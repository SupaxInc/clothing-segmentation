import scipy.io
import numpy as np
from PIL import Image
import os

def convert_mat_to_png(mat_directory, output_directory):
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

cwd = os.getcwd()  # Current directory

# Grab the absolute path for images, annotations, and masks
images_path = os.path.abspath(os.path.join(cwd, '../data/train/images/')) 
labels_path = os.path.abspath(os.path.join(cwd, '../data/annotations/pixel-level/'))
masks_path = os.path.abspath(os.path.join(cwd, '../data/train/masks/'))

convert_mat_to_png(labels_path, masks_path)