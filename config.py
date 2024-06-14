import torch

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-06
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 24
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