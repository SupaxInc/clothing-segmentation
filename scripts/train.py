import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from models.UNet import *
from scripts.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# TODO: Confirm these hyperparameters don't have possible underfitting or overfitting and are efficient
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
NUM_CLASSES = 7
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
    Training for one epoch. A batch of data will be processed per epoch.
    Args:
        loader: DataLoader that provides batches of the data (input images and masks) for training
        model: NN model that will be used for training
        optimizer: Updates the weights of the network based on gradients calculated during backpropogation
        loss_fn: Measures how well the model performs. Computes discrepancies between predicted outputs and targets (masks)
        scaler: Used for automatic mixed precision (AMP) to accelerate training while maintaining accuracy during forward/backward pass
    """
    batch = tqdm(loader) # Progress bar for the total amount of data

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
    # Train data set transformations (learning phase)
        # Normalizing pixel values, converting to tensor, and resizing to correct width and height
        # Adding randomness with rotations and flips to introduce variability and randomness
            # Helps promote robustness to variations in input data, mirroring a how a model would process new data
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    # Validation data set transformations (evaluation phase)
        # Normalizing pixel values, converting to tensor, and resizing to correct width and height
        # No randomness, ensures performance metrics from eval is indicative of how model performs on unseen data
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
    # Categorical Cross Entropy Loss for multi-class classifications
        # This will apply both cross-entropy loss and softmax activation in a single step
        # Helps prevent numerical instability and is a lot more efficient when done in a single step 
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_CLASSES,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    check_accuracy(val_loader, model, NUM_CLASSES, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    # An epoch is one complete pass through the entire dataset
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy
        val_loss = check_accuracy(val_loader, model, NUM_CLASSES, device=DEVICE)

        # Step the scheduler with the current epoch's validation loss
        scheduler.step(val_loss)

        # Print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="data/output_images/predictions/", device=DEVICE
        )

if __name__ == "__main__":
    main()