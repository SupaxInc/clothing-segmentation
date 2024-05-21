import scipy.io
import numpy as np
from PIL import Image
import os

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

cwd = os.getcwd()

# Grab the absolute path for images, annotations, and masks
    # Can custome these paths
images_path = os.path.abspath(os.path.join(cwd, '../data/train/images/')) 
labels_path = os.path.abspath(os.path.join(cwd, '../data/annotations/pixel-level/'))
masks_path = os.path.abspath(os.path.join(cwd, '../data/train/masks/'))

convert_mat_to_png(labels_path, masks_path)