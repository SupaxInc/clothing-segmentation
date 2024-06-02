import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ClothingCoParsingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample. Uses albumentations.
        """
        # Create a list of paths to all .jpg and .mat files within the relevant directories
        self.image_paths = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir)) if x.endswith('.jpg')]
        self.mask_paths = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir)) if x.endswith('.png')]

        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB')) # Converts PIL image to numpy array, each pixel will be 0 to 255
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'), dtype=np.uint8)  # Convert to grayscale, pixels 0 to 255 (whiteness)
        # Convert scaled mask back to original class labels
        mask = np.round(mask / 36).astype(np.uint8)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # One-hot encode mask for multi-class classifications, makes it compatible for cross entropy loss
            # Transforms categorical integer labels into binary matrix format to delineate which class a pixel belongs to
            # it needs to be more precise using floating point numbers, helps calculations for loss functions and gradients
        one_hot_mask = np.eye(self.num_classes, dtype=np.float32)[mask]
        # Moving the last axis (classes) to the first
            # Essentially changing shape to [num_classes, height, width]
            # E.g. (512, 512, 5), 512 x 512 image with 5 classes -> (5, 512, 512)
        one_hot_mask = np.moveaxis(one_hot_mask, -1, 0)

        return image, one_hot_mask
