import os
import sys
import cv2
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.file import get_paths_recursive
from src.config import SKY_FINDER_IMAGES_PATH, SKY_FINDER_SKY_CLASSES

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SkyFinderDataset(Dataset):
    """
    TODO: Add a description of the dataset.
    """

    def _get_paths_dict(self):
        """
        Initialize the paths dictionary for the dataset.
        The dictionary contains a 3-level structure:
        - Level 1: Sky class, either clear, partial or overcast
        - Level 2: Folder name, corresponding to the sky finder dataset camera identifier
        - Level 3: List of image paths
        """
        paths_dict = {}
        for sky_class in SKY_FINDER_SKY_CLASSES:
            paths_dict[sky_class] = {}
            sky_finder_class_images_path = f"{SKY_FINDER_IMAGES_PATH}/{sky_class}"
            folder_names = os.listdir(sky_finder_class_images_path)
            for folder_name in folder_names:
                folder_path = os.path.join(sky_finder_class_images_path, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                image_paths = get_paths_recursive(
                    folder_path=folder_path,
                    match_pattern="*.jpg",
                    path_type="f",
                    recursive=True,
                )
                paths_dict[sky_class][folder_name] = image_paths

        return paths_dict

    def _create_balanced_indices(self):
        # Get all sky classes and their camera IDs
        self.sky_classes = list(self.paths_dict.keys())
        self.camera_ids_by_class = {
            sky_class: list(self.paths_dict[sky_class].keys())
            for sky_class in self.sky_classes
        }

        # Get the number of images in total, per class, and per camera
        total_images = 0
        class_counts = {cls: 0 for cls in self.sky_classes}
        camera_counts = {cls: {} for cls in self.sky_classes}
        for sky_class in self.sky_classes:
            for camera_id in self.camera_ids_by_class[sky_class]:
                camera_images = self.paths_dict[sky_class][camera_id]
                image_count = len(camera_images)
                class_counts[sky_class] += image_count
                camera_counts[sky_class][camera_id] = image_count
                total_images += image_count

        # Add images to the dataset with corresponding weights
        self.all_images = []
        self.weights = []
        for sky_class in self.sky_classes:
            for camera_id in self.camera_ids_by_class[sky_class]:
                class_weight = 1.0 / (class_counts[sky_class] / total_images)
                camera_weight = 1.0 / (
                    camera_counts[sky_class][camera_id] / class_counts[sky_class]
                )

                camera_images = self.paths_dict[sky_class][camera_id]
                for img_path in camera_images:
                    self.all_images.append((sky_class, camera_id, img_path))
                    weight = class_weight * camera_weight
                    self.weights.append(weight)

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        # Define an epoch length
        n_combinations = sum(
            len(cameras) for cameras in self.camera_ids_by_class.values()
        )
        self.epoch_size = int(self.epoch_multiplier * n_combinations)

        print(f"✅ Created weighted dataset with {len(self.all_images)} total images.")
        print(f"🟢 Class distribution:")
        for cls in self.sky_classes:
            print(f"🟢 \t- {cls}: {class_counts[cls]} images.")
        print(f"🟢 Balanced epoch size: {self.epoch_size} samples.")

    def __init__(self, epoch_multiplier: int) -> None:
        """
        Initialize the SkyFinderDataset class.

        Args:
            epoch_multiplier (int): The average number of images to sample from a single class-camera pair in each epoch.
        """
        self.epoch_multiplier = epoch_multiplier
        self.paths_dict = self._get_paths_dict()
        self._create_balanced_indices()

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return self.epoch_size
    
    def _get_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __getitem__(self, idx):
        # Sample a random image from the dataset based on the weights
        sampled_idx = random.choices(range(len(self.all_images)), weights=self.weights, k=1)[0]
        sky_class, camera_id, img_path = self.all_images[sampled_idx]

        # Load the image
        image = self._get_image(img_path)

        return image
