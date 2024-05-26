import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ClothingCoParsingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Create a list of paths to all .jpg and .mat files within the relevant directories
        self.image_paths = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir)) if x.endswith('.jpg')]
        self.mask_paths = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir)) if x.endswith('.png')]

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB')) # Converts PIL image to numpy array, each pixel will be 0 to 255
        # Masks will be used as class identifiers, it needs to be more precise using floating point numbers
            # Helps calculations for loss functions and gradients
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'), dtype=np.float32)  # Convert to grayscale and float, pixels 0.0 to 255.0 (whiteness)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask