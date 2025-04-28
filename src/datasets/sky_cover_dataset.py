import os
import re
import sys
import cv2
import random
import numpy as np
import albumentations as A
import lightning.pytorch as pl
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.random import set_seed
from src.utils.random import SeededDataLoader
from src.utils.file import get_paths_recursive
from src.config import (
    SKY_COVER_PATH,
    SKY_COVER_WIDTH,
    SKY_COVER_HEIGHT,
    SKY_COVER_MAX_GROUND_TRUTH_VALUE,
)


class SkyCoverDataset(Dataset):
    """
    Sky cover dataset for training pixel-wise regression models.
    """

    def _get_data(
        self,
        paths: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Get the paths of the images and their corresponding ground truths.

        Args:
            paths (List[str]): List of paths to the dataset folders.

        Returns:
            List[str]: List of image paths.
            List[str]: List of ground truth paths.
        """
        all_images = []
        all_ground_truths = []

        for path in paths:
            data = get_paths_recursive(
                folder_path=path,
                match_pattern="*.png",
                path_type="f",
                recursive=False,
            )

            images = [f for f in data if re.match(r".*\d{2}\.png$", f)]
            images = sorted(images, key=lambda x: int(x.split("/")[-1].split(".")[0]))
            ground_truths = [f for f in data if re.match(r".*\d{2}_seg\.png$", f)]
            ground_truths = sorted(
                ground_truths, key=lambda x: int(x.split("/")[-1].split("_")[0])
            )

            all_images.extend(images)
            all_ground_truths.extend(ground_truths)

        return all_images, all_ground_truths

    def __init__(
        self,
        paths: List[str],
        use_augmentations: bool,
    ) -> None:
        super(SkyCoverDataset, self).__init__()

        self.paths = paths
        self.use_augmentations = use_augmentations

        self.all_images, self.all_ground_truths = self._get_data(paths)

        if use_augmentations:
            self.image_transform = A.Compose(
                [
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.2,
                        p=1.0,
                    ),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                    A.GaussNoise(
                        var_limit=(10.0, 50.0),
                        p=0.5,
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
        else:
            self.image_transform = A.Compose(
                [
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

        self.ground_truth_transform = A.Compose(
            [
                ToTensorV2(
                    p=1.0,
                ),
            ]
        )

        if use_augmentations:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(
                        p=0.5,
                    ),
                    A.Rotate(
                        limit=(-20, 20),
                        p=1.0,
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.Resize(height=SKY_COVER_HEIGHT, width=SKY_COVER_WIDTH, p=1.0),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=SKY_COVER_HEIGHT, width=SKY_COVER_WIDTH, p=1.0),
                ]
            )

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.all_images)

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

    def _get_ground_truth(
        self,
        ground_truth_path: str,
        max_value: int = SKY_COVER_MAX_GROUND_TRUTH_VALUE,
    ) -> np.ndarray:
        """
        Load a ground truth image from the given path and convert it to RGB format.

        Args:
            ground_truth_path (str): The path to the ground truth image.

        Returns:
            np.ndarray: The loaded ground truth image in RGB format.
        """
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(
                f"❌ Ground truth file not found: {os.path.abspath(ground_truth_path)}."
            )

        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth is None:
            raise FileNotFoundError(
                f"❌ Failed to load ground truth from: {os.path.abspath(ground_truth_path)}."
            )

        # Normalize based on the maximum value
        ground_truth = ground_truth.astype(np.float32) / float(max_value)
        ground_truth = np.clip(ground_truth, 0, 1)
        ground_truth = ground_truth * 255.0
        ground_truth = ground_truth.astype(np.uint8)

        return ground_truth

    def __getitem__(self, idx: int):
        image_path = self.all_images[idx]
        ground_truth_path = self.all_ground_truths[idx]

        # Load image and ground truth
        image = self._get_image(image_path)
        ground_truth = self._get_ground_truth(ground_truth_path)

        # Apply transformations
        transformed = self.transform(image=image, mask=ground_truth)
        image = transformed["image"]
        ground_truth = transformed["mask"]
        image = self.image_transform(image=image)["image"]
        ground_truth = self.ground_truth_transform(image=ground_truth)["image"]
        ground_truth = ground_truth / 255.0

        return image, ground_truth


class SkyCoverModule(pl.LightningDataModule):
    def _get_splitted_paths() -> Tuple[List[str], List[str]]:
        # Get folders for each sky type
        train_clear_paths = get_paths_recursive(
            folder_path=f"{SKY_COVER_PATH}clear/train",
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        val_clear_paths = get_paths_recursive(
            folder_path=f"{SKY_COVER_PATH}clear/val",
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        train_partial_paths = get_paths_recursive(
            folder_path=f"{SKY_COVER_PATH}partial/train",
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        val_partial_paths = get_paths_recursive(
            folder_path=f"{SKY_COVER_PATH}partial/val",
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        train_overcast_paths = get_paths_recursive(
            folder_path=f"{SKY_COVER_PATH}overcast/train",
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        val_overcast_paths = get_paths_recursive(
            folder_path=f"{SKY_COVER_PATH}overcast/val",
            match_pattern="*",
            path_type="d",
            recursive=False,
        )

        train_paths = train_clear_paths + train_partial_paths + train_overcast_paths
        val_paths = val_clear_paths + val_partial_paths + val_overcast_paths

        return train_paths, val_paths


    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        seed: Optional[int] = None,
    ) -> None:
        super(SkyCoverModule, self).__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        train_paths, val_paths = SkyCoverModule._get_splitted_paths()
        self.train_paths = train_paths
        self.val_paths = val_paths

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
            self.train_dataset = SkyCoverDataset(
                paths=self.train_paths,
                use_augmentations=True,
            )
            self.val_dataset = SkyCoverDataset(
                paths=self.val_paths,
                use_augmentations=False,
            )
        if stage == "test" or stage is None:
            pass

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

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the test dataloader.
        """
        return None
