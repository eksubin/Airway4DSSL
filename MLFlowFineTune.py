# %%
# Import libraries
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import mlflow

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    Resized,
    ScaleIntensityd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    LambdaD
)
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

#print_config()

# %%
# Command-line argument parsing

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train UNETR model with MONAI")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_iterations", type=int,
                        default=10000, help="Maximum number of iterations")
    parser.add_argument("--eval_num", type=int, default=100,
                        help="Number of iterations between evaluations")
    parser.add_argument("--experiment_name", type=str,
                        default="Test1", help="Name of the experiment")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--dropout", type=float,
                        default=0.0, help="Dropout rate")
    parser.add_argument("--use_pretrained", default=True,action="store_true",
                        help="Flag to use pretrained model")
    parser.add_argument("--pretrained_path", type=str,
                        default="./logs/Train-Thresh-255-2000EP-16th.pth", help="Path to the pretrained model")
    parser.add_argument("--train_dir", type=str,
                        default="./Data/FrenchSpeakerDataset/Segmentations/", help="Directory with training data")
    parser.add_argument("--val_dir", type=str, default="./Data/FrenchSpeakerDataset/Segmentations_Val/",
                        help="Directory with validation data")
    parser.add_argument("--note", type=str, default="",
                        help="Provide note about the specific run")
    return parser.parse_args()


args = parse_arguments()

# %%
# Setup parameters and device
lr = args.lr
max_iterations = args.max_iterations
eval_num = args.eval_num
experiment_name = args.experiment_name
epochs = args.epochs
dropout = args.dropout
use_pretrained = args.use_pretrained
pretrained_path = args.pretrained_path
train_dir = args.train_dir
val_dir = args.val_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for processing {device}")

# %%
# Setup log directory
logdir = os.path.normpath(f"./logs/{experiment_name}/")
if not os.path.exists(logdir):
    os.mkdir(logdir)

# %%
# Convert train and validation images into lists with locations
train_nrrd_files = sorted([os.path.join(train_dir, f) for f in os.listdir(
    train_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
train_seg_nrrd_files = sorted([os.path.join(train_dir, f)
                              for f in os.listdir(train_dir) if f.endswith(".seg.nrrd")])
val_nrrd_files = sorted([os.path.join(val_dir, f) for f in os.listdir(
    val_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
val_seg_nrrd_files = sorted([os.path.join(val_dir, f)
                            for f in os.listdir(val_dir) if f.endswith(".seg.nrrd")])

train_datalist = [{"image": img, "label": lbl}
                  for img, lbl in zip(train_nrrd_files, train_seg_nrrd_files)]
validation_datalist = [{"image": img, "label": lbl}
                       for img, lbl in zip(val_nrrd_files, val_seg_nrrd_files)]

# %%
# Define transforms for training and validation


def binarize_label(label):
    return (label > 0).astype(label.dtype)


def threshold_image(image):
    return np.where(image < 0.08, 0, image)


train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    LambdaD(keys="label", func=binarize_label),
    LambdaD(keys="image", func=threshold_image),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
    RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(
        64, 64, 64), random_size=False, num_samples=2),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    LambdaD(keys="label", func=binarize_label),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
    ToTensord(keys=["image", "label"]),
])

# %%
# Create DataLoaders for training and validation
train_ds = CacheDataset(data=train_datalist, transform=train_transforms,
                        cache_num=24, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=1,
                          shuffle=True, num_workers=4, pin_memory=True)

val_ds = CacheDataset(data=validation_datalist, transform=val_transforms,
                      cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True)

# %%
# Define and setup the model
model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(64, 64, 64),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="conv",
    norm_name="instance",
    res_block=True,
    dropout_rate=dropout,
)
model = model.to(device)

# Load pretrained weights if specified
if use_pretrained:
    print(f"Loading Weights from the Path {pretrained_path}")
    vit_dict = torch.load(pretrained_path, weights_only=True)
    vit_weights = vit_dict["state_dict"]
    model_dict = model.vit.state_dict()
    vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
    model_dict.update(vit_weights)
    model.vit.load_state_dict(model_dict)
    print("Pretrained Weights Successfully Loaded!")
else:
    print("No weights were loaded, all weights are randomly initialized!")

model.to(device)

# %%
# Setup loss function, optimizer, and metrics
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True,
                         reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# %%
# %% Define the validation component


def validation(epoch_iterator_val, dice_val_best):
    model.eval()
    dice_vals = []

    with torch.no_grad():
        for _step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (
                batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(
                val_inputs, (64, 64, 64), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(
                val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor)
                                  for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice))

        dice_metric.reset()

    mean_dice_val = np.mean(dice_vals)
    if mean_dice_val > dice_val_best:
        print(f"validation output shapes {val_outputs.shape}")
        image_slice = val_outputs[0, 0, :, :, 35].cpu().numpy() > 0.2
        # image_slice = (image_slice * 255).astype(np.uint8)
        image = Image.fromarray(image_slice)
        image_path = os.path.join(
            logdir, str(mean_dice_val) + experiment_name+ "_output_slice.png")
        image.save(image_path)
        mlflow.log_artifact(image_path)


       # Convert Pillow image to a matplotlib figure
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Turn off the axis

        # Save the figure to a temporary file and log it as an MLflow figure
        figure_path = os.path.join(logdir, experiment_name +
                                "_output_slice_figure.png")
        fig.savefig(figure_path, bbox_inches='tight', pad_inches=0)

        # Log the figure as an MLflow figure
        mlflow.log_artifact(figure_path)

    # Log validation dice score
    mlflow.log_metric('val_dice', mean_dice_val, step=global_step)

    return mean_dice_val
    #%% Define training function
from PIL import Image

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        # Log metric loss
        mlflow.log_metric('train_loss', loss.item(),
                          step=global_step)  # Log training loss

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val, dice_val_best)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                mlflow.log_metric('Current_step', global_step_best)
                # Log best dice value
                mlflow.log_metric('dice_val_best', dice_val_best)
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val))
                torch.save(model.state_dict(),
                           os.path.join(logdir, experiment_name + ".pth"))
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )

        global_step += 1
    return global_step, dice_val_best, global_step_best

    # %%
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
    mlflow.log_param('global_step_best', global_step_best)
    mlflow.log_param('Dataset', "0-1")
    mlflow.log_param('Dropout', dropout)
    mlflow.log_param('Descript', "corrected the validation to 0-255")

    # Run the program
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best)
    
    # Load the best model state
    model.load_state_dict(torch.load(
        os.path.join(logdir, experiment_name + ".pth")))

    # Log final model to MLflow
    mlflow.pytorch.log_model(model, f'models/{experiment_name}_final')

print(
    f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")
