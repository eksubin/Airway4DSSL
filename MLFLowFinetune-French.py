# %%
# Load the libraries
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

print_config()

# %%
# Setup the parameters
# Training Hyper-params
lr = 1e-4
max_iterations = 10000
eval_num = 100
experiment_name = "Test1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for processing {device}")
#device = torch.device("cpu")

# %%
#setup the log directory
logdir = os.path.normpath("./logs/Experiment1/")

if os.path.exists(logdir) is False:
    os.mkdir(logdir)

# Load the pre-trained model
use_pretrained = True
pretrained_path = os.path.normpath("./logs/Train-Thresh-255-2000EP-16th.pth")


# %%
# Convert the train and validation images into a list with locations
train_dir = "./Data/FrenchSpeakerDataset/Segmentations/"
val_dir = "./Data/FrenchSpeakerDataset/Segmentations_Val/"

# train image file
# timage_filenames = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith("im")])
train_nrrd_files = sorted([os.path.join(train_dir, f) for f in os.listdir(
    train_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
train_seg_nrrd_files = sorted([os.path.join(train_dir, f)
                               for f in os.listdir(train_dir) if f.endswith(".seg.nrrd")])
# validation image files
val_nrrd_files = sorted([os.path.join(val_dir, f) for f in os.listdir(
    val_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
val_seg_nrrd_files = sorted([os.path.join(val_dir, f)
                             for f in os.listdir(val_dir) if f.endswith(".seg.nrrd")])

# Create a list of dictionaries containing the file paths
train_datalist = [{"image": img, "label": lbl}
                  for img, lbl in zip(train_nrrd_files, train_seg_nrrd_files)]
validation_datalist = [{"image": img, "label": lbl}
                       for img, lbl in zip(val_nrrd_files, val_seg_nrrd_files)]

# Print the datalist to verify
print(validation_datalist[0])

# %%
# Binarizing function
def binarize_label(label):
    return (label > 0).astype(label.dtype)

#Thresholding function
def threshold_image(image):
    return np.where(image < 20, 0, image)


# Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"], minv=0, maxv=255),
        LambdaD(keys="label", func=binarize_label),
        LambdaD(keys="image", func=threshold_image),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
        RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(
            64, 64, 64), random_size=False, num_samples=2),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)

# Validation transforms
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        LambdaD(keys="label", func=binarize_label),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)


# %%
#Dataloaders

train_ds = CacheDataset(
    data=train_datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=2,
)
train_loader = DataLoader(train_ds, batch_size=1,
                          shuffle=True, num_workers=4, pin_memory=True)
val_ds = CacheDataset(data=validation_datalist, transform=val_transforms,
                      cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True)
print(f"Memmory summary {torch.cuda.memory_summary()}")

# %%
# Network 
# current GPU cannot handle this

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
    dropout_rate=0.0,
)
model = model.to(device)

# Load ViT backbone weights into UNETR
if use_pretrained is True:
    print("Loading Weights from the Path {}".format(pretrained_path))
    vit_dict = torch.load(pretrained_path)
    vit_weights = vit_dict["state_dict"]
    model_dict = model.vit.state_dict()

    vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
    model_dict.update(vit_weights)
    model.vit.load_state_dict(model_dict)
    del model_dict, vit_weights, vit_dict
    print("Pretrained Weights Succesfully Loaded !")

elif use_pretrained is False:
    print("No weights were loaded, all weights being used are randomly initialized!")

model.to(device)

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
def validation(epoch_iterator_val):
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
    # Log validation dice score
    mlflow.log_metric('val_dice', mean_dice_val, step=global_step)

    return mean_dice_val


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

        #Log metric loss
        mlflow.log_metric('train_loss', loss.item(),
                  step=global_step)  # Log training loss

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val))
                torch.save(model.state_dict(),
                           os.path.join(logdir, experiment_name + ".pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val)
                )
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
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
# Check if the experiment already exists
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    # If the experiment does not exist, create it
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    # If the experiment exists, get its ID
    experiment_id = experiment.experiment_id
# Set the experiment as the active experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    while global_step < max_iterations:
        mlflow.log_param('start_val', 1.0)
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best)

    # Load the best model state
    model.load_state_dict(torch.load(
        os.path.join(logdir, experiment_name + ".pth")))

    # Log final model to MLflow
    #mlflow.pytorch.log_model(model, f'models/{experiment_name}_final')

    # Optionally log other information
    mlflow.log_param('max_iterations', max_iterations)
    mlflow.log_param('global_step_best', global_step_best)
    mlflow.log_metric('dice_val_best', dice_val_best)

print(
    f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


# %%



