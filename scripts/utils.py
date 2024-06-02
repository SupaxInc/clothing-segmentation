import torch
import torchvision
import shutil
import os
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

    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Gradient calculation not needed during model eval
        for x, y in loader:
            # Batch of inputs (x) and targets (y) loaded to the device
            x = x.to(device) # Input images
            y = y.to(device) # Target masks
            
            outputs = model(x) # Pass the batch of input images to get raw scores (logits)
            preds = torch.argmax(outputs, dim=1) # For each pixel, it selects the class with the highest score (predicted class)
            num_correct += (preds == y).sum() # How many predicted class labels match the mask labels
            num_pixels += torch.numel(preds) # Total num of preds made (or pixels evaluated)

            # Calculate the dice score for each class and average
                # The dice score is a measure of overlap between the predicted and true masks, averaged over all classes.
            for class_idx in range(num_classes):
                # Creates a boolean tensor for the predicted class and the true mask class
                    # Float converts it to a tensor of 0.0 or 1.0. 
                    # This tensor represents all pixels predicted to belong to current class
                pred_i = (preds == class_idx).float()
                true_i = (y == class_idx).float()

                intersection = (pred_i * true_i).sum() # Area where the prediced class and the true masks agree (true positives)
                
                # We then measure the overlap between the predicted class and the true mask class using Dice formula
                    # The factor of 2 ensures that the score ranges from 0 (no overlap) to 1 (perfect overlap)
                dice_score += (2 * intersection) / (
                    # The sum of the predicted class and the true mask class
                    # Adding 1e-8 to avoid division by zero
                    pred_i.sum() + true_i.sum() + 1e-8
                )

    dice_score /= num_classes * len(loader) # Average Dice score over all classes and batches

    print(
        f"Got {num_correct}/{num_pixels} with accuary {num_correct/num_pixels*100:.2f}%"
    )
    print(f"Average dice score: {dice_score}")

    model.train()

def save_predictions_as_imgs(
    loader, model, folder="data/images/predictions", device="cuda"
):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
        
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