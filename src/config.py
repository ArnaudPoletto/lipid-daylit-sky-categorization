import os
import sys
GLOBAL_PATH = str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) + "/"

DATA_PATH = f"{GLOBAL_PATH}data/"
SKY_FINDER_PATH = f"{DATA_PATH}sky_finder/"
SKY_FINDER_IMAGES_PATH = f"{SKY_FINDER_PATH}images/"
SKY_FINDER_MASKS_PATH = f"{SKY_FINDER_PATH}masks/"
SKY_FINDER_SKY_CLASSES = ["clear", "partial", "overcast"]

PATCH_WIDTH = 640
PATCH_HEIGHT = 360
