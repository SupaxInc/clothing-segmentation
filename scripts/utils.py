import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from config import (CLASS_MAPPING, CLASS_MAPPING_NAMING)
from data.dataset import *
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Used to save the current state of the model, including model weights and optimizer states.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Loads a previously saved checkpoint into the model. Used to resume training or for inference using trained parameters.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"]) # Load the model weights from a checkpoint file

def move_files(files, source, destination):
        if not os.path.exists(destination):
            os.makedirs(destination)
        for f in files:
            shutil.copy(os.path.join(source, f), destination)

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_classes,
    num_workers=4,
    pin_memory=True, # If pin_memory true, data loader will copy tensors into CUDA pinned memory before returning them
):
    """
    Datasets will contain the input images and target masks. It will then be provided to dataloaders to be used by the model in batches.
    """
    # Create dataset instances for training
        # Datasets will contain the input images and target masks that we can unpack with the dataloader.
    train_ds = ClothingCoParsingDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        num_classes=num_classes,
        transform=train_transform,
    )

    # DataLoader for the training dataset
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        shuffle=True, # Shuffling for randomness in training
    )

    # Create dataset instance for validation
    val_ds = ClothingCoParsingDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        num_classes=num_classes,
        transform=val_transform,
    )

    # DataLoader for the validation dataset
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def dice_loss(preds, targets, smooth=1.):
    # preds: [B, C, H, W]
    # targets: [B, C, H, W] (one-hot encoded)
    
    preds = F.softmax(preds, dim=1)
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(preds, targets):
    # preds: [B, C, H, W]
    # targets: [B, C, H, W] (one-hot encoded)

    # For CrossEntropyLoss, we need class indices
    target_indices = torch.argmax(targets, dim=1)

    # Categorical Cross Entropy Loss for multi-class classifications, handle class imbalance
        # This will apply both cross-entropy loss and softmax activation in a single step
        # Helps prevent numerical instability and is a lot more efficient when done in a single step 
        # Focuses on pixel-wise classification accuracy
    ce_loss = F.cross_entropy(preds, target_indices)

    # Focuses on the overlap between the predicted segmentation and the ground truth
        # Effective for boundary accuracy especially segmentation
    dl_loss = dice_loss(preds, targets)

    return ce_loss + dl_loss

def check_accuracy(loader, model, num_classes, device="cuda"):
    """
    Calculates the accuracy by comparing the predicted class labels against the true labels across the entire dataset.
    It also computes the Dice score, which is a measure of overlap between the predicted and true segmentation masks, averaged over all classes.
    Args:
        loader (DataLoader): The loader contains input images (x) and target masks (y) data that was loaded using the dataset.
        model (torch.nn.Module): The model that is used for prediction.
        num_classes (int): The number of classes in the dataset.
        device (str): The device type, 'cuda' or 'cpu', where the computations will be performed.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_loss = 0

    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Gradient calculation not needed during model eval
        for x, y in loader:
            # Batch of inputs (x) and targets (y) loaded to the device
            x = x.to(device) # Input images
            y = y.long().to(device) # Target masks (one-hot encoded)
            
            outputs = model(x) # Pass the batch of input images to get raw scores (logits)
            preds = torch.argmax(outputs, dim=1) # For each pixel, it selects the class with the highest score (predicted class)
            y_indices = torch.argmax(y, dim=1) # Convert one-hot encoded masks back to class indices for comparison

            num_correct += (preds == y_indices).sum() # How many predicted class labels match the mask labels
            num_pixels += torch.numel(preds) # Total num of preds made (or pixels evaluated)

            loss = combined_loss(outputs, y)
            total_loss += loss.item()

            # Calculate the dice score for each class and average
                # The dice score is a measure of overlap between the predicted and true masks, averaged over all classes.
            for class_idx in range(num_classes):
                # Creates a boolean tensor for the predicted class and the true mask class
                    # Float converts it to a tensor of 0.0 or 1.0. 
                    # This tensor represents all pixels predicted to belong to current class
                pred_i = (preds == class_idx).float()
                true_i = (y_indices == class_idx).float()

                intersection = (pred_i * true_i).sum() # Area where the prediced class and the true masks agree (true positives)
                
                # We then measure the overlap between the predicted class and the true mask class using Dice formula
                    # The factor of 2 ensures that the score ranges from 0 (no overlap) to 1 (perfect overlap)
                dice_score += (2 * intersection) / (
                    # The sum of the predicted class and the true mask class
                    # Adding 1e-8 to avoid division by zero
                    pred_i.sum() + true_i.sum() + 1e-8
                )

    dice_score /= num_classes * len(loader) # Average Dice score over all classes and batches
    avg_loss = total_loss / len(loader)

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}%"
    )
    print(f"Average dice score: {dice_score}")
    print(f"Average validation loss: {avg_loss}")

    model.train()
    return avg_loss

def save_predictions_as_imgs(
    loader, model, folder="data/images/predictions", device="cuda"
):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

        # Converting one-hot encoded masks as class indices before saving image
            # Ensures that saved images represent class labels rather than one-hot encoded vectors, which would not be meaningful as images.
        y = torch.argmax(y, dim=1)
        
        # Save predicted masks and ground truth masks
        torchvision.utils.save_image(
            # Normalization for better visualization so that class preds represent masks correctly
            # Add channel dimension with unsqueeze so it has correct tensor shape [B, C, H, W] for saving as image
            preds.unsqueeze(1).float() / preds.max(),
            f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            # Similarly to masks, we also normalize and add channel dimensions for ground truths
            y.unsqueeze(1).float() / y.max(), 
            f"{folder}/gt_{idx}.png"
        )

    model.train()

def show_result(model, x, y, title='Result', save_file=False, save_file_name='show_result', device="cuda"):
    model.eval()
    
    x = x.to(device)
    with torch.no_grad():
        outputs = model(x) # [B, C, H, W]
        preds = torch.argmax(outputs, dim=1) # [B, H, W]

    y = torch.argmax(y, dim=1) # [B, num_classes, H, W] -> [B, H, W]

    # Convert pred image and mask to proper shape for matplotlib
    preds = preds.squeeze(0).unsqueeze(2) # [B, H, W] -> [H, W, 1]
    y = y.squeeze(0).unsqueeze(2) # [B, H, W] -> [H, W, 1]

    # Convert input image to proper shape for matplot lib
    # Just incase, move image tensor to CPU since matplotlib does not support GPU tensors
    x_cpu = x.cpu().numpy().squeeze(0).transpose(1, 2, 0) # Change shape from [B, C, H, W] -> [C, H, W] -> [H, W, C] for RGB

    fig = plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=25)

    # For image plot
    plt.subplot(1, 4, 1)
    plt.imshow(x_cpu)
    plt.title('Image')

    # For ground truth plot
    plt.subplot(1, 4, 2)
    plt.imshow(y.cpu().numpy(), cmap='gray') # Change mask to a numpy array and visualized as gray scale
    plt.title('Ground truth')

    # For prediction plot
    plt.subplot(1, 4, 3)
    plt.imshow(preds.cpu().numpy(), cmap='gray')
    plt.title('Prediction')

    # For overlay plot that contains image and mask
    plt.subplot(1, 4, 4)
    plt.imshow(x_cpu)
    # Create a masked array where the mask is applied to locations where the prediction is 0 (background)
    masked_imclass = np.ma.masked_where(preds.cpu().numpy() == 0, preds.cpu().numpy())
    plt.imshow(masked_imclass, alpha=0.4, cmap='jet')
    plt.title('Prediction over Image')

    plt.show()

    if save_file:
        save_dir = 'results/show_result/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + save_file_name)

def show_segmentation(model, x, title='Class Segmentation', save_file=False, save_file_name='segmentation', device="cuda"):
    x = x.to(device)
    with torch.no_grad():
        outputs = model(x)  # [B, C, H, W]
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]

    # Convert input image to proper shape for matplotlib
    x_cpu = x.cpu().numpy().squeeze(0).transpose(1, 2, 0)  # [H, W, C = 3 for RGB] 

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=25)

    # Plot original image
    axes[0, 0].imshow(x_cpu)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].imshow(x_cpu)
    axes[0, 0].axis('off') # Remove the axis ticks and labels for a cleaner look
    
    # Plot each class segmentation
    for i, (class_id, _) in enumerate(CLASS_MAPPING.items()):
        # Calculate the row and column for the subplot. +1 because 0,0 is taken by original image
        row, col = divmod(i+1, 4)
        
        # Create a mask where True values are where the prediction is equal to the current class
            # Converts to numpy and removes the batch dimension from preds
        mask = preds.cpu().numpy().squeeze(0) == i
        
        # Create a copy of the original image
        masked_class = x_cpu.copy()
        
        # Apply the mask to the image copy
            # ~mask selects all pixels that are not part of the current class, so we set those that are False pixel values to True
        masked_class[~mask] = [0, 0, 0]  # Allows us to set non-class pixels to black
        
        # Plot the masked image on the corresponding subplot
        axes[row, col].imshow(masked_class)
        axes[row, col].set_title(f'Class {i}: {CLASS_MAPPING_NAMING[class_id]}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    if save_file:
        save_dir = 'results/show_result/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(os.path.join(save_dir, save_file_name))
