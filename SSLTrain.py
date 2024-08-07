# %%
import os
import json
import time
import torch
import matplotlib.pyplot as plt
from torch.nn import DataParallel

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    ScaleIntensityd,
    LambdaD
)
import numpy as np
import argparse
import mlflow

#%%


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train UNETR model with MONAI")
    parser.add_argument("--max_iterations", type=int,
                        default=1000, help="Maximum number of iterations")
    parser.add_argument("--image_max", type=float, default=1,
                        help="Max_image_intensity")
    parser.add_argument("--image_min", type=float, default=0,
                        help="Min_image_intensity")
    parser.add_argument("--threshold", type=float, default=0.08,
                        help="Min_image_intensity")
    parser.add_argument("--model_name", type=str, default="",
                        help="provide the model name")
    parser.add_argument("--note", type=str, default="",
                        help="provide the model name")
    parser.add_argument("--dataset", type=str, default="0-1",
                        help="provide dataset type")

    return parser.parse_args()


args = parse_arguments()

image_max = args.image_max
image_min = args.image_min
max_iterations = args.max_iterations
model_name = args.model_name
threshold = args.threshold
note = args.note
dataset = args.dataset

# %%
logdir_path = os.path.normpath("./logs/")

#Convert the train and validation images into a list with locations
train_dir = "./Data/FrenchSpeakerDataset/NRRD_Files_N4Bias/"
val_dir = "./Data/FrenchSpeakerDataset/NRRD_Files_N4Bias_Val/"

#train image file
timage_filenames = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".nrrd")])

#validation image files
vimage_filenames = sorted([os.path.join(val_dir, f)
                          for f in os.listdir(val_dir) if f.endswith(".nrrd")])

# Create a list of dictionaries containing the file paths
train_datalist = [{"image": img} for img in timage_filenames]
validation_datalist = [{"image": img} for img in vimage_filenames]

# Print the datalist to verify
print(train_datalist[0], validation_datalist[0])

# %%
# Define Training Transforms
def threshold_image(image):
    return np.where(image < threshold, 0, image)

train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Defines the image intensity
        ScaleIntensityd(keys=["image"], minv=image_min, maxv=image_max),
        # Defines threshold for the image
        LambdaD(keys="image", func=threshold_image),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(64, 64, 64)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(
            64, 64, 64), random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=2, names=[
            "gt_image", "image_2"], allow_missing_keys=False),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8,
                           holes=10, spatial_size=8),
    ]
)

# %%
# Training Config

# Define Network ViT backbone & Loss & Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTAutoEnc(
    in_channels=1,
    img_size=(64, 64, 64),
    patch_size=(16, 16, 16),
    proj_type="conv",
    hidden_size=768,
    mlp_dim=3072,
)

model = model.to(device)

if torch.cuda.device_count() > 1:

    model = DataParallel(model)
    print(f"###### Using data parallism {torch.cuda.device_count()}")

# Define Hyper-paramters for training loop
experiment_name = model_name
max_epochs = max_iterations
val_interval = 2
batch_size = 4
lr = 1e-4
epoch_loss_values = []
step_loss_values = []
epoch_cl_loss_values = []
epoch_recon_loss_values = []
val_loss_values = []
best_val_loss = 1000.0

recon_loss = L1Loss()
contrastive_loss = ContrastiveLoss(temperature=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


# Define DataLoader using MONAI, CacheDataset needs to be used
train_ds = Dataset(data=train_datalist, transform=train_transforms)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_ds = Dataset(data=validation_datalist, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=True, num_workers=1)

# %%  MLFLow tracking

# Start MLflow run
mlflow.set_tracking_uri("file:./mlruns")

# Check if the experiment already exists
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Optionally log other information
    mlflow.log_param('max_iterations', max_iterations)
    mlflow.log_param('Dataset', dataset)
    mlflow.log_param('note', note)

# %%
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        mlflow.log_metric("epoch", epoch)
        model.train()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            start_time = time.time()

            inputs, inputs_2, gt_input = (
                batch_data["image"].to(device),
                batch_data["image_2"].to(device),
                batch_data["gt_image"].to(device),
            )
            optimizer.zero_grad()
            outputs_v1, hidden_v1 = model(inputs)
            outputs_v2, hidden_v2 = model(inputs_2)

            flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
            flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

            r_loss = recon_loss(outputs_v1, gt_input)
            cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

            # Adjust the CL loss by Recon Loss
            total_loss = r_loss + cl_loss * r_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            step_loss_values.append(total_loss.item())

            # CL & Recon Loss Storage of Value
            epoch_cl_loss += cl_loss.item()
            epoch_recon_loss += r_loss.item()

            end_time = time.time()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {total_loss.item():.4f}, "
                f"time taken: {end_time-start_time}s"
            )
            mlflow.log_metric('train_loss',total_loss.item())

        epoch_loss /= step
        epoch_cl_loss /= step
        epoch_recon_loss /= step

        epoch_loss_values.append(epoch_loss)
        epoch_cl_loss_values.append(epoch_cl_loss)
        epoch_recon_loss_values.append(epoch_recon_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch % val_interval == 0:
            print("Entering Validation for epoch: {}".format(epoch + 1))
            total_val_loss = 0
            val_step = 0
            model.eval()
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                inputs, gt_input = (
                    val_batch["image"].to(device),
                    val_batch["gt_image"].to(device),
                )
                print("Input shape: {}".format(inputs.shape))
                outputs, outputs_v2 = model(inputs)
                val_loss = recon_loss(outputs, gt_input)
                total_val_loss += val_loss.item()
                end_time = time.time()

            total_val_loss /= val_step
            val_loss_values.append(total_val_loss)
            print(
                f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

            if total_val_loss < best_val_loss:
                print(
                    f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(
                ), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(logdir_path, experiment_name + ".pth"))
                mlflow.log_metric("best_val_loss", total_val_loss)
                mlflow.pytorch.log_model(model, f'models/{experiment_name}_final')

    print("Done")


# %%



