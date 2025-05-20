import os
import sys
import cv2
import torch
import numpy as np
import albumentations as A
import lightning.pytorch as pl
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.random import set_seed
from src.utils.random import SeededDataLoader
from src.transformations.mean_patches import MeanPatches
from src.utils.sky_finder import (
    get_sky_finder_masks,
    get_sky_finder_bounding_boxes,
    get_splitted_sky_finder_paths_dict,
)
from src.config import (
    EPOCH_MULTIPLIERS,
    SKY_FINDER_WIDTH,
    SKY_FINDER_HEIGHT,
    N_PAIRS,
)


class ContrastivePairsDataset(Dataset):
    """
    Contrastive pairs dataset.
    """

    def _create_balanced_indices(
        self,
        epoch_multiplier: int,
    ) -> Tuple[List[Tuple[str, str, str]], List[float], int]:
        """
        Create a balanced dataset with weights for each image based on the class and camera ID.

        Args:
            epoch_multiplier (int): The average number of images to sample from a single class-camera pair in each epoch.

        Returns:
            List[Tuple[str, str, str]]: A list of tuples containing the class, camera ID, and image path.
            List[float]: A list of weights for each image.
            int: The size of the epoch.
        """
        # Get all sky classes and their camera IDs
        sky_classes = list(self.paths_dict.keys())
        camera_ids_by_class = {
            sky_class: list(self.paths_dict[sky_class].keys())
            for sky_class in sky_classes
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

    def __init__(
        self,
        paths_dict: Dict[str, Dict[str, List[str]]],
        epoch_multiplier: int,
        n_pairs: int = 3,
    ) -> None:
        """
        Initialize the contrastive pairs dataset.

        Args:
            epoch_multiplier (int): The average number of images to sample from a single class-camera pair in each epoch.
            n_pairs (int): The number of pairs to sample for each image.
        """
        super(ContrastivePairsDataset, self).__init__()

        self.paths_dict = paths_dict
        self.epoch_multiplier = epoch_multiplier
        self.n_pairs = n_pairs

        self.all_images, self.weights, self.epoch_size = self._create_balanced_indices(
            epoch_multiplier
        )
        self.masks = get_sky_finder_masks(paths_dict)
        self.bounding_boxes = get_sky_finder_bounding_boxes(paths_dict)

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
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"❌ Image file not found: {os.path.abspath(image_path)}."
            )

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"❌ Failed to load image from: {os.path.abspath(image_path)}."
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _get_patch(image: np.ndarray) -> np.ndarray:
        """
        Get a patch from the image with random transformations.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The transformed patch.
        """
        transform = A.Compose(
            [
                A.HorizontalFlip(
                    p=0.5,
                ),
                A.Rotate(
                    limit=(-20, 20),
                    p=1.0,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.Resize(height=SKY_FINDER_HEIGHT, width=SKY_FINDER_WIDTH, p=1.0),
                A.ImageCompression(quality_range=(50, 100), p=0.5),
                A.CLAHE(clip_limit=2, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.GaussNoise(std_range=(0.0, 0.1), p=0.3),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=0.5),
                        A.MedianBlur(blur_limit=5, p=0.5),
                        A.GaussianBlur(blur_limit=5, p=0.5),
                    ],
                    p=0.3,
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    p=0.5,
                ),
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=0.5,
                ),
                MeanPatches(
                    num_patches=(2, 5), patch_size=(50, 150), use_image_mean=True, p=0.7
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(
                    p=1.0,
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            torch.Tensor: The processed image patch pair.
        """
        # Get n_pairs image indexes, without replacement
        sampled_idx = np.random.choice(
            range(len(self.all_images)),
            size=self.n_pairs,
            replace=False,
            p=self.weights,
        )
        processed_images = [
            self._get_processed_image_from_idx(idx) for idx in sampled_idx
        ]
        pairs = [
            torch.stack(
                [
                    ContrastivePairsDataset._get_patch(image),
                    ContrastivePairsDataset._get_patch(image),
                ],
                dim=0,
            )
            for image in processed_images
        ]
        pairs = torch.stack(pairs, dim=0)

        return pairs


class ContrastivePairsModule(pl.LightningDataModule):
    """
    Contrastive pairs data module.
    """

    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the contrastive pairs data module.

        Args:
            batch_size (int): The batch size for the dataloaders.
            n_workers (int): The number of workers for the dataloaders.
            seed (Optional[int]): The seed for random number generation.
        """
        super(ContrastivePairsModule, self).__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        train_paths_dict, val_paths_dict, test_paths_dict = (
            get_splitted_sky_finder_paths_dict()
        )
        self.train_paths_dict = train_paths_dict
        self.val_paths_dict = val_paths_dict
        self.test_paths_dict = test_paths_dict

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation and testing.

        Args:
            stage (Optional[str]): The stage of the training process. Can be "fit", "validate", "test" or None.
        """
        if self.seed is not None:
            print(f"🌱 Setting the seed to {self.seed} for generating dataloaders.")
            set_seed(self.seed)

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ContrastivePairsDataset(
                paths_dict=self.train_paths_dict,
                epoch_multiplier=EPOCH_MULTIPLIERS[0],
                n_pairs=N_PAIRS,
            )
            self.val_dataset = ContrastivePairsDataset(
                paths_dict=self.val_paths_dict,
                epoch_multiplier=EPOCH_MULTIPLIERS[1],
                n_pairs=N_PAIRS,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ContrastivePairsDataset(
                paths_dict=self.test_paths_dict,
                epoch_multiplier=EPOCH_MULTIPLIERS[2],
                n_pairs=N_PAIRS,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        """
        return SeededDataLoader(
            self.val_dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
            DataLoader: The test dataloader.
        """
        return SeededDataLoader(
            self.test_dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
        )
