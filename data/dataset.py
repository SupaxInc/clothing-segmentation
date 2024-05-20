import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

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
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask