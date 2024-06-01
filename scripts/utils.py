import torch
import torchvision
from ..data.dataset import ClothingCoParsingDataset
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

# TODO: Ensure that training and validation datasets are processed in pre_processing.py
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True, # If pin_memory true, data loader will copy tensors into CUDA pinned memory before returning them
):
    """
    Creates dataloaders for training and validation datasets
    """
    # Create dataset instances for training
    train_ds = ClothingCoParsingDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
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
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad(): # Gradient calculation not needed during model eval
        for x, y in loader:
            # Batch of inputs (x) and targets (y) loaded to the device
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x) # Raw scores (logits)
            preds = torch.argmax(outputs, dim=1) # For each pixel, it selects the class with the highest score (predicted class)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # Calculate the dice score for each class and average
            for class_idx in range(num_classes):
                pred_i = (preds == class_idx).float()
                true_i = (y == class_idx).float()
                intersection = (pred_i * true_i).sum()
                dice_score += (2 * intersection) / (
                    pred_i.sum() + true_i.sum() + 1e-8
                )

    dice_score /= num_classes * len(loader)

    print(
        f"Got {num_correct}/{num_pixels} with accuary {num_correct/num_pixels*100:.2f}%"
    )
    print(f"Average dice score: {dice_score}")

    model.train()

# TODO: Convert this for multi-class classification
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()