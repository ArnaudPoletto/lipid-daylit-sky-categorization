import os
import sys
import torch

SEED = 0

GLOBAL_PATH = str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) + "/"
DATA_PATH = f"{GLOBAL_PATH}data/"
MODELS_PATH = f"{DATA_PATH}models/"

SKY_FINDER_PATH = f"{DATA_PATH}sky_finder/"
SKY_FINDER_IMAGES_PATH = f"{SKY_FINDER_PATH}images/"
SKY_FINDER_MASKS_PATH = f"{SKY_FINDER_PATH}masks/"
SKY_FINDER_SKY_CLASSES = ["clear", "partial", "overcast"]

CHECKPOINT_PATH = f"{MODELS_PATH}contrastive_net/20250414-205233_contrastive_net/epoch=3-val_loss=1.08.ckpt"

PATCH_WIDTH = 640
PATCH_HEIGHT = 360
N_PAIRS = 3
PROJECTION_DIM = 16

N_EPOCHS = 10
BATCH_SIZE = 2
N_WORKERS = 8
EVALUATION_STEPS = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SPLITS = (0.8, 0.1, 0.1)
EPOCH_MULTIPLIERS = (100, 10, 10)
CRITERION_TEMPERATURE = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
