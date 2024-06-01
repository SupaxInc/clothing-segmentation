import scipy.io
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from utils import move_files

def convert_mat_to_png(mat_directory, output_directory):
    """
    Converts mat files as png, each pixel in the PNG will correspond to a class label indicated
    by the groundtruth annotation data so it can be used for training.
    Args:
        mat_directory (string): Directory with all the dataset mat files
        output_directory (string): Directory to save the mat files as png for training
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(mat_directory):
        if filename.endswith('.mat'):
            mat_path = os.path.join(mat_directory, filename)
            data = scipy.io.loadmat(mat_path)
            annotation = data['groundtruth']

            # Convert annotation matrix to an Image
            img = Image.fromarray(annotation.astype(np.uint8))
            img.save(os.path.join(output_directory, filename.replace('.mat', '.png')))
    
def split_data(base_image_path, base_mask_path, train_size=0.8):
    """
    Split the data into train and test sets
    Args:
        base_image_path (string): Path to the directory with the images
        base_mask_path (string): Path to the directory with the masks
        train_size (float): Fraction of the data to use for training, default 80%
    """
    # Create a list of all files in the directories
    image_files = [img for img in os.listdir(base_image_path) if img.endswith('.jpg')]
    mask_files = [mask for mask in os.listdir(base_mask_path) if mask.endswith('.png')]

    # Sort the data to ensure that the images and masks are aligned
    image_files.sort()
    mask_files.sort()

    # Split the data into train and test sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, train_size=train_size, random_state=42
    )

    train_images_path = os.path.join(base_image_path, '../../train/images')
    train_masks_path = os.path.join(base_mask_path, '../../train/masks')
    val_images_path = os.path.join(base_image_path, '../../validation/images')
    val_masks_path = os.path.join(base_mask_path, '../../validation/masks')

    move_files(train_images, base_image_path, train_images_path)
    move_files(train_masks, base_mask_path, train_masks_path)
    move_files(val_images, base_image_path, val_images_path)
    move_files(val_masks, base_mask_path, val_masks_path)

cwd = os.getcwd()

# Grab the absolute path for images, annotations, and masks
    # Can customize these paths
images_path = os.path.abspath(os.path.join(cwd, '../data/input_images/images/')) 
labels_path = os.path.abspath(os.path.join(cwd, '../data/annotations/pixel-level/'))
masks_path = os.path.abspath(os.path.join(cwd, '../data/input_images/masks/'))

# Converts the pixel level annotated mats to pngs
convert_mat_to_png(labels_path, masks_path)

# Split the data into train and test sets
split_data(images_path, masks_path)