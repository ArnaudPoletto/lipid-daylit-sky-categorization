import os
import torch
import numpy as np

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_PATH = str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) + "/"
DATA_PATH = f"{GLOBAL_PATH}data/"
GENERATED_PATH = f"{GLOBAL_PATH}generated/"
MODELS_PATH = f"{DATA_PATH}models/"

# Texture descriptor
SKY_FINDER_PATH = f"{DATA_PATH}sky_finder/"
SKY_FINDER_IMAGES_PATH = f"{SKY_FINDER_PATH}images/"
SKY_FINDER_MASKS_PATH = f"{SKY_FINDER_PATH}masks/"
SKY_FINDER_ARCHIVES_PATH = f"{SKY_FINDER_PATH}archives/"
SKY_FINDER_EXTRACTED_PATH = f"{SKY_FINDER_PATH}extracted/"
KSY_FINDER_CATEGORY_MAPPING_FILE_PATH = f"{SKY_FINDER_PATH}category_mapping.json"
EMBEDDINGS_FILE_PATH = f"{GENERATED_PATH}embeddings.json"
EMBEDDINGS_PLOT_FILE_PATH = f"{GENERATED_PATH}embeddings_plot.png"
CONTRASTIVE_CHECKPOINT_PATH = f"{MODELS_PATH}contrastive_net/baseline.ckpt"

SKY_FINDER_TRAIN_SPLIT = 0.6
SKY_FINDER_VAL_SPLIT = 0.2
SKY_FINDER_TEST_SPLIT = 0.2
assert np.isclose(SKY_FINDER_TRAIN_SPLIT + SKY_FINDER_VAL_SPLIT + SKY_FINDER_TEST_SPLIT, 1.0), "Train, val and test splits must sum to 1."

SKY_FINDER_CAMERA_IDS = [65, 75, 162, 623, 858, 3297, 3395, 3396, 4232, 4584, 5020, 5021, 7371, 8733, 8953, 10066, 11160, 17218, 19388, 19834]
SKY_FINDER_SKY_CLASSES = ["clear", "partial", "overcast"]

SKY_FINDER_WIDTH = 640
SKY_FINDER_HEIGHT = 360
N_PAIRS = 3
PROJECTION_DIM = 16

EPOCH_MULTIPLIERS = (100, 10, 10)
CRITERION_TEMPERATURE = 0.5

# Cloud coverage estimator
SKY_FINDER_COVER_PATH = f"{DATA_PATH}sky_finder_cover/"
SKY_FINDER_ACTIVE_PATH = f"{DATA_PATH}sky_finder_active_learning/"
UNET_MANUAL_CHECKPOINT_PATH = f"{MODELS_PATH}unet/baseline_manual.ckpt"
UNET_ACTIVE_CHECKPOINT_PATH = f"{MODELS_PATH}unet/baseline_active.ckpt"

SKY_COVER_MAX_GROUND_TRUTH_VALUE = 224

# Classification
SKY_FINDER_DESCRIPTORS_PATH = f"{GENERATED_PATH}sky_finder_descriptors.json"
ALL_SKY_CLASS_NET_CHECKPOINT_PATH = f"{MODELS_PATH}sky_class_net/all_baseline.ckpt"
CONTRASTIVE_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH = f"{MODELS_PATH}sky_class_net/contrastive_only_baseline.ckpt"
COVER_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH = f"{MODELS_PATH}sky_class_net/cover_only_baseline.ckpt"

# Pipeline
PROCESSED_VIDEOS_PATH = f"{DATA_PATH}videos/processed/"
GENERATED_PIPELINE_PATH = f"{GENERATED_PATH}pipeline/"