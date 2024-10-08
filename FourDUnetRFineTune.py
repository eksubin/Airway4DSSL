# %%
# Import libraries
# %%
# Import libraries
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import mlflow
from torch.nn import DataParallel
from PIL import Image

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    LambdaD,
    ConcatItemsd,
    RandRotate90d,
    ToTensord
)
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)
#print_config() #Configuration output(optional)

# %%
# Command-line argument parsing

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train UNETR model with MONAI")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum number of iterations")
    parser.add_argument("--eval_num", type=int, default=2, help="Number of iterations between evaluations")
    parser.add_argument("--experiment_name", type=str, default="Test1", help="Name of the experiment")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--use_pretrained", action="store_true", help="Flag to use pretrained model")
    parser.add_argument("--pretrained_path", type=str, default="./logs/Train-Thresh-255-2000EP-16th.pth", help="Path to the pretrained model")
    parser.add_argument("--train_dir", type=str, default="", help="Directory with training data")
    parser.add_argument("--val_dir", type=str, default="", help="Directory with validation data")
    parser.add_argument("--note", type=str, default="", help="Provide note about the specific run")
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
note = args.note
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup log directory
logdir = os.path.normpath(f"./logs/{experiment_name}/")
os.makedirs(logdir, exist_ok=True)

# %%
# Create train and validation dataset

def create_multichannel_datalist(image_filenames, label_filenames, stack_size=5):
    datalist = []
    end = len(image_filenames) - stack_size + 1
    start = 0
    while start <= end:
        img_dict = {}
        for j in range(stack_size):
            img_dict[f"image{j}"] = str(image_filenames[start + j])
        for j in range(stack_size):
            img_dict[f"label{j}"] = str(label_filenames[start + j])
        datalist.append(img_dict)
        start += stack_size
    return datalist

# Gather training and validation image paths
train_nrrd_files = sorted([
    os.path.join(train_dir, f) for f in os.listdir(train_dir)
    if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")
])
train_seg_nrrd_files = sorted([
    os.path.join(train_dir, f) for f in os.listdir(train_dir)
    if f.endswith(".seg.nrrd")
])

val_nrrd_files = sorted([
    os.path.join(val_dir, f) for f in os.listdir(val_dir)
    if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")
])
val_seg_nrrd_files = sorted([
    os.path.join(val_dir, f) for f in os.listdir(val_dir)
    if f.endswith(".seg.nrrd")
])

# Create data lists for training and validation
train_datalist = create_multichannel_datalist(train_nrrd_files, train_seg_nrrd_files)
validation_datalist = create_multichannel_datalist(train_nrrd_files[:5], train_seg_nrrd_files[:5])
print(f"Training datalist setup: {train_datalist[0]}")

# %%
# Define transforms for training and validation

def binarize_label(label):
    # Binarize the label: convert to 1 if > 0, else 0
    return (label > 0).astype(label.dtype)

def threshold_image(image):
    # Threshold the image: set values below 0.08 to 0
    return np.where(image < 0.08, 0, image)

def invert_image(image):
    # Binarize the label: convert to 1 if > 0, else 0
    return 1-image

# Training transforms
train_transforms = Compose([
    LoadImaged(keys=["image0", "image1", "image2", "image3", "image4", "label0","label1","label2","label3","label4"]),
    EnsureChannelFirstd(keys=["image0", "image1", "image2", "image3", "image4", "label0","label1","label2","label3","label4"]),
    ConcatItemsd(keys=["image0", "image1", "image2", "image3", "image4"], name="image"),
    ConcatItemsd(keys=["label0","label1","label2","label3","label4"], name="label"),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    LambdaD(keys="label", func=binarize_label),
    LambdaD(keys="image", func=threshold_image),
    LambdaD(keys="image", func=invert_image),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
    RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(64, 64, 64), random_size=False, num_samples=1),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ToTensord(keys=["image", "label"]),
])

# Validation transforms
val_transforms = Compose([
    LoadImaged(keys=["image0", "image1", "image2", "image3", "image4", "label0", "label1", "label2", "label3", "label4"]),
    EnsureChannelFirstd(keys=["image0", "image1", "image2", "image3", "image4", "label0", "label1", "label2", "label3", "label4"]),
    ConcatItemsd(keys=["image0", "image1", "image2", "image3", "image4"], name="image"),
    ConcatItemsd(keys=["label0", "label1", "label2", "label3", "label4"], name="label"),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    LambdaD(keys="label", func=binarize_label),
    LambdaD(keys="image", func=threshold_image),
    LambdaD(keys="image", func=invert_image),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
    # RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(64, 64, 64), random_size=False, num_samples=1),
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

################################ modified Unet
from typing import Tuple, Union
import torch

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT

class UNETR4D(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        pos_embed: str,
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            #stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            #stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            #stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            #stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, X):
        outputs = []
        encStack = []
        for t in range(X.size(1)):
            #print(f" in size {X.size()}")
            x_in = X[:,t,:,:,:].unsqueeze(0)
            #print(f"xin shape{x_in.size()}")
            x, hidden_states_out = self.vit(x_in)
            enc1 = self.encoder1(x_in)
            x2 = hidden_states_out[3]
            enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
            x3 = hidden_states_out[6]
            enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
            x4 = hidden_states_out[9]
            enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
            #encoder stack
            #print(f" enc4 shape {enc4.size()}")
            ###########################################change this if it is required
            # encStack.append(enc4)
            # if t > 0:
            #     enc4 = torch.cat((enc4,encStack[t]), dim=1)
            #     enc4 = enc4.view(1, 128, 2, 8, 8, 8).sum(dim=2)
            ########################################################################
            dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
            dec3 = self.decoder5(dec4, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)
            logits = self.out(out)
            outputs.append(logits)
        return torch.stack(outputs, dim=2)

model = UNETR4D(
    in_channels=1,
    out_channels=1,
    img_size=(64, 64,64),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="conv",
    norm_name="instance",
    res_block=True,
)
model = model.to(device)
################################

###############################This is the actual model
# model = UNETR(
#     in_channels=5,
#     out_channels=5,
#     img_size=(64, 64, 64),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="conv",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=dropout,
# )
# model = model.to(device)
###################################################################


# model parallelization
# if torch.cuda.device_count() > 1:

#     model = DataParallel(model)
#     print(f"###### Using data parallism {torch.cuda.device_count()}")


# Load pretrained weights if specified
if use_pretrained:
    print(f"Loading weights from: {pretrained_path}")
    vit_dict = torch.load(pretrained_path, weights_only=False)
    vit_weights = model.state_dict()
    model_dict = model.vit.state_dict()
    vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
    model_dict.update(vit_weights)
    model.vit.load_state_dict(model_dict)
    print("Pretrained weights successfully loaded!")
else:
    print("No weights loaded; weights initialized randomly!")

# %%
# Setup loss function, optimizer, and metrics

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

post_label = AsDiscrete()
post_pred = AsDiscrete()
dice_metric = DiceMetric(include_background=True,
                         reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# %%
# Define the validation component
def custom_sliding_window_inference(inputs, model, roi_size, overlap=0.25):
    """
    Performs custom sliding window inference.

    Args:
        inputs (torch.Tensor): Input tensor of shape (N, C, D, H, W).
        model (torch.nn.Module): The model for inference.
        roi_size (tuple): The size of the region of interest (D, H, W).
        overlap (float): The overlap ratio between sliding windows.

    Returns:
        torch.Tensor: Inference results.
    """
    D, H, W = inputs.shape[2], inputs.shape[3], inputs.shape[4]
    stride = [int(roi_size[0] * (1 - overlap)), int(roi_size[1] * (1 - overlap)), int(roi_size[2] * (1 - overlap))]

    pad_depth = max(0, roi_size[0] - D)
    pad_height = max(0, roi_size[1] - H)
    pad_width = max(0, roi_size[2] - W)

    # Padding: pad dimensions to handle edge cases
    inputs_padded = torch.nn.functional.pad(inputs, (0, pad_width, 0, pad_height, 0, pad_depth), mode='replicate')

    output_shape = (inputs.shape[0], inputs.shape[1], D, H, W)
    outputs = torch.zeros(output_shape, device=inputs.device)

    counts = torch.zeros(output_shape, device=inputs.device)

    for z in range(0, D, stride[0]):
        for y in range(0, H, stride[1]):
            for x in range(0, W, stride[2]):
                # Define the window slice
                z_start = min(z, D)
                y_start = min(y, H)
                x_start = min(x, W)

                z_end = min(z + roi_size[0], D + pad_depth)
                y_end = min(y + roi_size[1], H + pad_height)
                x_end = min(x + roi_size[2], W + pad_width)

                # Extract the region of interest
                window = inputs_padded[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                #print(f"window size {window.size()}")
                # Predict using the model
                with torch.no_grad():
                    prediction = model(window)
                #print(f"output shape {prediction.squeeze(0).size()}")
                #print(f"outputs shape {outputs.size()}")
                # Accumulate results
                outputs[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += prediction.squeeze(0)
                counts[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += 1
    # Normalize the predictions by count
    outputs /= counts.clamp(min=1)  # Avoid division by zero
    return outputs

def validation(epoch_iterator_val, dice_val_best):
    model.eval()
    dice_vals = []

    with torch.no_grad():
        for _step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
            # if val_labels.shape[1] != 1:  # Reduce to first channel if needed
            #     val_labels = val_labels[:, 0, ...] + val_labels[:, 1, ...]
            #     val_labels = val_labels.unsqueeze(1)
            #print(f"val_inputs {val_inputs.size()} val labels {val_labels.size()}")


            # Replace existing sliding window inference code with the custom function
            val_outputs = custom_sliding_window_inference(
                inputs=val_inputs,
                model=model,
                roi_size=(64, 64, 64),
                overlap=0
            )

            #print(f"val_inputs {val_inputs.size()} val labels {val_labels.size()}, val output {val_outputs.size()}")
            # val_labels_list = decollate_batch(val_labels)
            # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            # val_outputs_list = decollate_batch(val_outputs)
            # val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            # Calculate the Dice metric for each channel and average
            dice_array = []
            for channel in range(val_outputs.size(1)):  # Loop through each channel
                # Reset the metric before calculating for each channel
                dice_metric.reset()

                # Compute the Dice metric for the current channel
                # Normalize the predictions to the range [0, 1]
                print(f"val out shape{val_outputs.size()}")
                # Normalize predictions to the range [0, 1]
                normalized_pred = (val_outputs[:, channel, :, :, :] - val_outputs.min()) / (val_outputs.max() - val_outputs.min())
                # Ensure normalization respects negative values
                normalized_pred = (normalized_pred - normalized_pred.min()) / (normalized_pred.max() - normalized_pred.min())
                # Apply threshold to get binary predictions
                pred = post_pred(normalized_pred)  # Predictions after thresholding

                true = post_label(val_labels[:, channel, :, :, :])  # Ground truth labels
                print(f"output min values {pred.min()}_{normalized_pred.min()}_{pred.max()}_{true.min()}")
                # Updating the Dice metric
                dice_metric(y_pred=pred, y=true)
                dice_channel = dice_metric.aggregate().item()

                #print(f"Dice Metric for channel {channel}: {dice_channel}")
                dice_array.append(dice_channel)

            # Calculate the mean Dice across all channels
            dice = np.mean(dice_array)
            dice_vals.append(dice)
            epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice))


    mean_dice_val = np.mean(dice_vals)
    if mean_dice_val > dice_val_best:
        np.save(os.path.join(logdir, f"{experiment_name}_{mean_dice_val}.npy"), val_outputs.cpu().numpy())
        np.save(os.path.join(logdir, f"{experiment_name}_{mean_dice_val}_label.npy"), val_labels.cpu().numpy())
        print("Numpy output is saved")


    mlflow.log_metric('val_dice', mean_dice_val, step=global_step)

    return mean_dice_val
#%%
# Define training function

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))

        # If labels have more than one channel, combine them into one
        # if y.shape[1] != 1:
        #     y = y[:, 0, ...] + y[:, 1, ...]
        #     y = y.unsqueeze(1)
        print(f"input shape {x.min(), x.max(), y.min(), y.max()}")
        logit_map = model(x)  # Forward pass
        #print(f"logit map size {logit_map[:,:,1,:,:,:].squeeze(0).size()} {y[:,1,:,:,:].size()}")
        # loss = loss_function(logit_map.squeeze(0), y)  ####################### actual logit function
        ##################################### updated loss
        loss = 0.0
        # Calculate the loss for each channel and average it
        losses = [loss_function(logit_map[:,:,channel,:,:,:], y[:,channel,:,:,:].unsqueeze(0)) for channel in range(logit_map.size(2))]
        loss = torch.mean(torch.stack(losses))  # Average the losses across channels
        ################################################################################
        loss.backward()  # Backpropagation
        epoch_loss += loss.item()
        optimizer.step()  # Update parameters
        optimizer.zero_grad()  # Reset gradients
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        # Log training loss
        mlflow.log_metric('train_loss', loss.item(), step=global_step)
        ## TODO
        # torch.save(model.state_dict(), os.path.join(logdir, experiment_name + ".pth"))
        # print("Model Saved! Best Avg. Dice")
        ######### remove this

        # Evaluate the model periodically
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val, dice_val_best)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            # Update best dice score and save model if improved
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                mlflow.log_metric('Current_step', global_step_best)
                mlflow.log_metric('dice_val_best', dice_val_best)
                print("Model Saved! Best Avg. Dice: {} Current Avg. Dice: {} Step {}".format(dice_val_best, dice_val, global_step))
                torch.save(model.state_dict(), os.path.join(logdir, experiment_name + ".pth"))
            else:
                print("Model Not Saved! Best Avg. Dice: {} Current Avg. Dice: {} Step {}".format(dice_val_best, dice_val, global_step))
        global_step += 1

    return global_step, dice_val_best, global_step_best

# %%
# Start MLflow run

mlflow.set_tracking_uri("file:./mlruns")

# Check if the experiment exists; create it if not
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Log essential parameters
    mlflow.log_param('max_iterations', max_iterations)
    mlflow.log_param('global_step_best', global_step_best)
    mlflow.log_param('Dataset', "0-1")
    mlflow.log_param('Dropout', dropout)
    mlflow.log_param('Descript', note)

    # Training loop
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best)

    # Load and log the best model
    model.load_state_dict(torch.load(os.path.join(logdir, experiment_name + ".pth")))
    mlflow.pytorch.log_model(model, f'models/{experiment_name}_final')

print(f"Training completed, best metric: {dice_val_best:.4f} at iteration: {global_step_best}")


