# TODO: make test and val more seeded: precompute list of indexes maybe? be aware of seeding transformations too maybe...

import os
import sys
import cv2
import random
import albumentations as A
from typing import Tuple, List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.file import get_paths_recursive
from src.config import (
    SKY_FINDER_IMAGES_PATH,
    SKY_FINDER_SKY_CLASSES,
    SKY_FINDER_MASKS_PATH,
    PATCH_WIDTH,
    PATCH_HEIGHT,
)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SkyFinderDataset(Dataset):
    """
    TODO: Add a description of the dataset.
    """

    def _create_balanced_indices(
        self, 
        epoch_multiplier: int
    ) -> Tuple[list, list, int]:
        """
        Create a balanced dataset with weights for each image based on the class and camera ID.

        Args:
            epoch_multiplier (int): The average number of images to sample from a single class-camera pair in each epoch.

        Returns:
            all_images (list): A list of tuples containing (sky_class, camera_id, image_path).
            weights (list): A list of weights for each image.
            epoch_size (int): The size of the epoch.
        """
        # Get all sky classes and their camera IDs
        sky_classes = list(self.paths_dict.keys())
        camera_ids_by_class = {
            sky_class: list(self.paths_dict[sky_class].keys()) for sky_class in sky_classes
        }

        # Get the number of images in total, per class, and per camera
        total_images = 0
        class_counts = {cls: 0 for cls in sky_classes}
        camera_counts = {cls: {} for cls in sky_classes}
        for sky_class in sky_classes:
            for camera_id in camera_ids_by_class[sky_class]:
                camera_images = self.paths_dict[sky_class][camera_id]
                image_count = len(camera_images)
                class_counts[sky_class] += image_count
                camera_counts[sky_class][camera_id] = image_count
                total_images += image_count

        # Add images to the dataset with corresponding weights
        all_images = []
        weights = []
        for sky_class in sky_classes:
            for camera_id in camera_ids_by_class[sky_class]:
                class_weight = 1.0 / (class_counts[sky_class] / total_images)
                camera_weight = 1.0 / (
                    camera_counts[sky_class][camera_id] / class_counts[sky_class]
                )

                camera_images = self.paths_dict[sky_class][camera_id]
                for img_path in camera_images:
                    all_images.append((sky_class, camera_id, img_path))
                    weight = class_weight * camera_weight
                    weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Define an epoch length
        n_combinations = sum(len(cameras) for cameras in camera_ids_by_class.values())
        epoch_size = int(epoch_multiplier * n_combinations)

        print(f"✅ Created weighted dataset with {len(all_images)} total images.")
        print(f"🟢 Class distribution:")
        for cls in sky_classes:
            print(f"🟢 \t- {cls}: {class_counts[cls]} images.")
        print(f"🟢 Balanced epoch size: {epoch_size} samples.")

        return all_images, weights, epoch_size

    def _get_camera_ids(self) -> List[str]:
        """
        Get the camera IDs from the dataset.

        Returns:
            List[str]: A list of camera IDs.
        """
        camera_ids = []
        for sky_class in self.paths_dict.keys():
            camera_ids.extend(self.paths_dict[sky_class].keys())
        unique_camera_ids = list(set(camera_ids))

        return unique_camera_ids

    def _get_masks(self) -> Dict[str, np.ndarray]:
        """
        Get the camera masks from the dataset, segmenting the sky from the ground.

        Returns:
                Dict[str, np.ndarray]: A dictionary with camera IDs as keys and masks as values.
        """
        camera_ids = self._get_camera_ids()
        masks = {}
        for camera_id in camera_ids:
            mask_path = f"{SKY_FINDER_MASKS_PATH}/{camera_id}.png"
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask > 128
                masks[camera_id] = mask
            else:
                print(f"❌ Mask not found for camera ID {camera_id}.")

        print(f"✅ Found {len(masks)} masks for {len(camera_ids)} camera IDs.")

        return masks

    def _get_bounding_boxes(
        masks: Dict[str, np.ndarray],
        ground_percentage_threshold: float = 0.1,
        min_height: int = 50,
        crop_step: int = 1,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Get the bounding boxes for the cameras based on the masks.

        Args:
            masks (Dict[str, np.ndarray]): A dictionary with camera IDs as keys and masks as values.
            ground_percentage_threshold (float): The threshold for the ground percentage.
            min_height (int): The minimum height for the bounding box.
            crop_step (int): The step size for the binary search.

        Returns:
            Dict[str, Tuple[int, int, int, int]]: A dictionary with camera IDs as keys and bounding boxes as values.
        """
        bounding_boxes = {}
        for camera_id, mask in masks.items():
            height, width = mask.shape
            y_min, y_max = 0, height

            # Start with full image
            ground_pixels = np.sum(~mask)
            total_pixels = width * height
            ground_percentage = ground_pixels / total_pixels
            while True:
                # Keep cropping until we're below threshold or can't crop anymore
                if ground_percentage <= ground_percentage_threshold:
                    break
                if (y_max - y_min) <= min_height:
                    break
                if y_min + crop_step >= y_max - crop_step:
                    break

                # Get top part
                top_rows = mask[y_min : y_min + crop_step, :]
                top_ground_pixels = np.sum(~top_rows)
                top_density = top_ground_pixels / (width * crop_step)
                # Get bottom part
                bottom_rows = mask[y_max - crop_step : y_max, :]
                bottom_ground_pixels = np.sum(~bottom_rows)
                bottom_density = bottom_ground_pixels / (width * crop_step)

                # Select the part with the most sky and update the bounding box
                total_pixels -= width * crop_step
                if top_density > bottom_density:
                    ground_pixels -= top_ground_pixels
                    y_min += crop_step
                else:
                    ground_pixels -= bottom_ground_pixels
                    y_max -= crop_step

                ground_percentage = ground_pixels / total_pixels

            # Set the bounding box
            bounding_boxes[camera_id] = (0, y_min, width, y_max)

        print(f"✅ Found bounding boxes for {len(bounding_boxes)} camera IDs.")

        return bounding_boxes

    def __init__(
        self,
        paths_dict: Dict[str, Dict[str, List[str]]],
        epoch_multiplier: int,
    ) -> None:
        """
        Initialize the SkyFinderDataset class.

        Args:
            epoch_multiplier (int): The average number of images to sample from a single class-camera pair in each epoch.
        """
        self.paths_dict = paths_dict
        self.all_images, self.weights, self.epoch_size = self._create_balanced_indices(epoch_multiplier)
        self.masks = self._get_masks()
        self.bounding_boxes = SkyFinderDataset._get_bounding_boxes(self.masks)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return self.epoch_size

    def _get_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from the given path and convert it to RGB format.

        Args:
            image_path (str): The path to the image.

        Returns:
            np.ndarray: The loaded image in RGB format.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _get_patch(
        image: np.ndarray,
        patch_ratio: float = PATCH_WIDTH / PATCH_HEIGHT,
    ) -> np.ndarray:
        """
        Get a patch from the image with random transformations.

        Args:
            image (np.ndarray): The input image.
            patch_ratio (float): The aspect ratio of the patch.

        Returns:
            np.ndarray: The transformed patch.
        """
        height, width = image.shape[:2]

        # Get patch size range
        corrected_width = min(int(height * patch_ratio), width)

        # Get padding transform
        pad_vertically = np.random.choice([True, False])
        if pad_vertically:
            padding_transform = A.PadIfNeeded(
                min_height=height + np.random.randint(0, height * 0.2 + 1),
                min_width=width,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            )
        else:
            padding_transform = A.PadIfNeeded(
                min_height=height,
                min_width=width + np.random.randint(0, width * 0.2 + 1),
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            )

        transform = A.Compose(
            [
                A.HorizontalFlip(
                    p=0.5,
                ),
                padding_transform,
                A.Rotate(
                    limit=(-20, 20),
                    p=1.0,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.RandomCrop(height=height, width=corrected_width, p=1.0),
                A.Resize(height=PATCH_HEIGHT, width=PATCH_WIDTH, p=1.0),
                A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
                A.CLAHE(clip_limit=(0, 1), p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0,
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.5,
                ),
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=0.5,
                ),
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.5,
                ),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                A.GaussNoise(
                    var_limit=(10.0, 50.0),
                    p=0.5,
                ),
            ]
        )

        # Apply the transformations
        transformed = transform(image=image)
        patch = transformed["image"]

        return patch

    def _get_processed_image_from_idx(
        self,
        idx: int,
    ) -> np.ndarray:
        """
        Get the processed image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            np.ndarray: The processed image.
        """
        _, camera_id, img_path = self.all_images[idx]

        # Load the image
        image = self._get_image(img_path)
        if camera_id not in self.masks:
            raise ValueError(f"❌ Mask not found for camera ID {camera_id}.")
        mask = self.masks[camera_id]

        # Crop image and mask based on bounding box
        x_min, y_min, x_max, y_max = self.bounding_boxes[camera_id]
        image = image[y_min:y_max, x_min:x_max]
        mask = mask[y_min:y_max, x_min:x_max]

        # Apply inpaining to fill the ground
        inpaint_mask = (~mask).astype(np.uint8) * 255
        filled_image = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)

        return filled_image

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the anchor patch, positive patch, and negative patch.
        """
        # Get anchor and positive patches
        sampled_idx = random.choices(
            range(len(self.all_images)), weights=self.weights, k=1
        )[0]
        processed_image = self._get_processed_image_from_idx(sampled_idx)
        anchor_patch = SkyFinderDataset._get_patch(processed_image)
        positive_patch = SkyFinderDataset._get_patch(processed_image)

        # Get negative patch
        neg_sampled_idx = sampled_idx
        while neg_sampled_idx == sampled_idx:
            neg_sampled_idx = random.choices(
                range(len(self.all_images)), weights=self.weights, k=1
            )[0]
        processed_negative_image = self._get_processed_image_from_idx(neg_sampled_idx)
        negative_patch = SkyFinderDataset._get_patch(processed_negative_image)

        return anchor_patch, positive_patch, negative_patch


def get_paths_dict() -> Dict[str, Dict[str, List[str]]]:
    """
    Initialize the paths dictionary for the dataset.
    The dictionary contains a 3-level structure:
    - Level 1: Sky class, either clear, partial or overcast
    - Level 2: Folder name, corresponding to the sky finder dataset camera identifier
    - Level 3: List of image paths

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary with sky classes as keys and dictionaries of folder names and image paths as values.
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


def get_splitted_paths_dict(
    train_split: float,
    val_split: float,
    test_split: float,
) -> Tuple[
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[str]]],
]:
    """
    Split the paths dictionary into train, validation and test sets.

    Args:
        train_split (float): The proportion of the dataset to include in the train split.
        val_split (float): The proportion of the dataset to include in the validation split.
        test_split (float): The proportion of the dataset to include in the test split.

    Returns:
        Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]: A tuple containing three dictionaries for train, validation and test splits.
    """
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("❌ Train, validation and test splits must sum to 1.0.")

    # Get the paths dictionary
    paths_dict = get_paths_dict()

    # Split the paths dictionary into train, validation and test sets
    train_paths_dict = {}
    val_paths_dict = {}
    test_paths_dict = {}
    for sky_class, folders in paths_dict.items():
        train_paths_dict[sky_class] = {}
        val_paths_dict[sky_class] = {}
        test_paths_dict[sky_class] = {}

        for folder_name, image_paths in folders.items():
            shuffled_image_paths = image_paths.copy()
            random.shuffle(shuffled_image_paths)

            n_images = len(shuffled_image_paths)
            n_train = int(n_images * train_split)
            n_val = int(n_images * val_split)

            train_paths_dict[sky_class][folder_name] = shuffled_image_paths[:n_train]
            val_paths_dict[sky_class][folder_name] = shuffled_image_paths[
                n_train : n_train + n_val
            ]
            test_paths_dict[sky_class][folder_name] = shuffled_image_paths[
                n_train + n_val :
            ]

    return train_paths_dict, val_paths_dict, test_paths_dict


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    epoch_multipliers: Tuple[int, int, int] = (100, 10, 10),
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create the DataLoaders for the train, validation and test sets.
    
    Args:
        batch_size (int): The batch size for the DataLoaders.
        num_workers (int): The number of workers for the DataLoaders.
        epoch_multiplier (int): The average number of images to sample from a single class-camera pair in each epoch.
        train_split (float): The proportion of the dataset to include in the train split.
        val_split (float): The proportion of the dataset to include in the validation split.
        test_split (float): The proportion of the dataset to include in the test split.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the train, validation and test DataLoaders.
    """
    if not np.isclose(sum(splits), 1.0):
        raise ValueError("❌ Train, validation and test splits must sum to 1.0.")
    
    train_paths_dict, val_paths_dict, test_paths_dict = get_splitted_paths_dict(
        train_split=splits[0],
        val_split=splits[1],
        test_split=splits[2],
    )

    train_dataset = SkyFinderDataset(
        paths_dict=train_paths_dict,
        epoch_multiplier=epoch_multipliers[0],
    )
    val_dataset = SkyFinderDataset(
        paths_dict=val_paths_dict,
        epoch_multiplier=epoch_multipliers[1],
    )
    test_dataset = SkyFinderDataset(
        paths_dict=test_paths_dict,
        epoch_multiplier=epoch_multipliers[2],
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"✅ Created DataLoaders with batch size {batch_size} and {num_workers} workers."
    )
    print(f"🟢 Train dataset size: {len(train_dataset)}.")
    print(f"🟢 Validation dataset size: {len(val_dataset)}.")
    print(f"🟢 Test dataset size: {len(test_dataset)}.")

    return train_dataloader, val_dataloader, test_dataloader
