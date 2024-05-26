import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from models import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
NUM_CLASSES = 5
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"
VAL_IMG_DIR = "data/validations/images/"
VAL_MASK_DIR = "data/validations/masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    One epoch training.
    Args:
        loader: DataLoader that provides batches of the data (input images and masks) for training
        model: NN model that will be used for training
        optimizer: Updates the weights of the network based on gradients calculated during backpropogation
        loss_fn: Measures how well the model performs. Computes discrepancies between predicted outputs and targets (masks)
        scaler: Used for automatic mixed precision (AMP) to accelerate training while maintaining accuracy during forward/backward pass
    """
    batch = tqdm(loader) # Progress bar for the amount of data


    for batch_idx, (data, targets) in enumerate(batch):
        data = data.to(device=DEVICE) # Assigning data to appropriate device (CPU or GPU)
        targets = targets.to(device=DEVICE) # Prepares data (may need to reshape data depending on what loss_fn uses)

        # Forward pass to generate predictions and calculate loss using autocasting
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets) # Find out the loss between predictions and targets

        # Backward pass to compute gradients and update models weights
        optimizer.zero_grad() # Zero gradients to prevent accumulation from previous iteration
        scaler.scale(loss).backward() # Scales the loss to prevent underflow during backpropogation
        scaler.step(optimizer) # Adjust weights based on calculated gradients 
        scaler.update() # Update scaler for next iteration based on if gradients overflowed

        # Update progress bar batch to show current loss
        batch.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss # Cross Entropy Loss for multi-class classifications
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # Print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    


if __name__ == "__main__":
    main()