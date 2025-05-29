import os
import re
import sys
import cv2
import torch
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
    SKY_FINDER_WIDTH,
    SKY_FINDER_PATH,
    SKY_FINDER_HEIGHT,
    SKY_FINDER_COVER_PATH,
    SKY_FINDER_ACTIVE_KEEP_PATH,
)


class SkyFinderCoverDataset(Dataset):
    """
    Sky Finder cover dataset.
    """

    def _get_data(
        self,
        path: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Get the paths of all images and ground truths in the dataset.

        Args:
            path (str): Path to the dataset.

        Returns:
            Tuple[List[str], List[str], List[str]]: Paths to all images and ground truths.

        """
        images_path = f"{path}/images"
        ground_truth_path = f"{path}/ground_truths"

        all_images = get_paths_recursive(
            folder_path=images_path,
            match_pattern="*.jpg",
            path_type="f",
            recursive=True,
        )
        all_images = sorted(all_images)
        all_ground_truths = get_paths_recursive(
            folder_path=ground_truth_path,
            match_pattern="*.jpg",
            path_type="f",
            recursive=True,
        )
        all_ground_truths = sorted(all_ground_truths)

        # Check missing images
        for ground_truth in all_ground_truths:
            image_path = re.sub(r"/ground_truths/", "/images/", ground_truth)
            if image_path not in all_images:
                raise ValueError(
                    f"❌ Image at path {os.path.abspath(image_path)} does not exist for ground truth at path {os.path.abspath(ground_truth)}."
                )
        # Check missing ground truths
        for image in all_images:
            ground_truth_path = re.sub(r"/images/", "/ground_truths/", image)
            if ground_truth_path not in all_ground_truths:
                raise ValueError(
                    f"❌ Cloud ground truth at path {os.path.abspath(ground_truth_path)} does not exist for image at path {os.path.abspath(image)}."
                )

        # Check if all ground truths have a corresponding image
        if len(all_images) != len(all_ground_truths):
            raise ValueError(
                f"❌ Number of images ({len(all_images)}) does not match number of ground truths ({len(all_ground_truths)})."
            )
        for ground_truth in all_ground_truths:
            image_path = re.sub(r"ground_truths", "images", ground_truth)
            if image_path not in all_images:
                raise ValueError(
                    f"❌ Image at path {os.path.abspath(image_path)} does not exist for ground truth at path {os.path.abspath(ground_truth)}."
                )

        return all_images, all_ground_truths

    def _get_pseudo_labelling_data(self) -> Tuple[List[str], List[str]]:
        """
        Get the paths of all images and ground truth images in the dataset.

        Returns:
            Tuple[List[str], List[str]]: List of image paths and ground truth image paths.
        """
        all_jpg_files = get_paths_recursive(
            folder_path=SKY_FINDER_ACTIVE_KEEP_PATH,
            match_pattern="*.jpg",
            path_type="f",
            recursive=True,
        )

        datetime_pattern = re.compile(r"/\d{8}_\d{6}\.jpg$")
        all_images = [path for path in all_jpg_files if datetime_pattern.search(path)]

        ground_truth_pattern = re.compile(r"/binary_\d{8}_\d{6}\.jpg$")
        ground_truth_images = [
            path for path in all_jpg_files if ground_truth_pattern.search(path)
        ]

        print(
            f"✅ Found {len(all_images)} images and {len(ground_truth_images)} ground truth images."
        )
        return all_images, ground_truth_images

    def __init__(
        self,
        path: str,
        use_augmentations: bool,
        with_pseudo_labelling: bool,
    ) -> None:
        """
        Initialize the SkyFinderCoverDataset class.

        Args:
            path (str): Path to the dataset.
            use_augmentations (bool): Whether to use augmentations.
            with_pseudo_labelling (bool): Whether to use pseudo labelling.
        """
        super(SkyFinderCoverDataset, self).__init__()

        self.path = path
        self.use_augmentations = use_augmentations
        self.with_pseudo_labelling = with_pseudo_labelling

        self.all_images, self.all_ground_truths = self._get_data(path)

        if with_pseudo_labelling:
            self.all_pl_images, self.all_pl_ground_truths = (
                self._get_pseudo_labelling_data()
            )
            self.all_images += self.all_pl_images
            self.all_ground_truths += self.all_pl_ground_truths

        if use_augmentations:
            self.image_transform = A.Compose(
                [
                    A.ImageCompression(quality_range=(50, 100), p=0.3),
                    A.CLAHE(clip_limit=2, p=0.3),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=0.5,
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
                    A.RandomFog(
                        fog_coef_range=(0.1, 0.3),
                        p=0.2,
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        src_radius=100,
                        src_color=(255, 255, 255),
                        angle_range=(0, 1),
                        num_flare_circles_range=(1, 3),
                        p=0.2,
                    ),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(p=1.0),
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
                    ToTensorV2(p=1.0),
                ]
            )

        self.ground_truth_transform = A.Compose(
            [
                ToTensorV2(p=1.0),
            ]
        )

        if use_augmentations:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(
                        limit=(-20, 20),
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.RandomResizedCrop(
                        size=(SKY_FINDER_HEIGHT, SKY_FINDER_WIDTH),
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        p=0.5,
                    ),
                    A.Resize(height=SKY_FINDER_HEIGHT, width=SKY_FINDER_WIDTH, p=1.0),
                ],
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=SKY_FINDER_HEIGHT, width=SKY_FINDER_WIDTH, p=1.0),
                ],
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

        # Get the mask and apply it to the image
        camera_id = int(image_path.split("/")[-2])
        mask_path = f"{SKY_FINDER_PATH}masks/{camera_id}.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        mask = mask > (255 * 0.5)
        mask = np.expand_dims(mask, axis=-1)
        image = np.where(mask, image, 0)

        return image

    def _get_ground_truth(
        self,
        ground_truth_path: str,
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

        return ground_truth

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image and ground truth tensors.
        """
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

        ground_truth = torch.where(ground_truth > 0.4, 1.0, 0.0).float()

        return image, ground_truth


class SkyFinderCoverModule(pl.LightningDataModule):
    """
    Sky Finder cover data module.
    """

    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        with_pseudo_labelling: bool,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the SkyFinderCoverModule class.

        Args:
            batch_size (int): Batch size for the dataloaders.
            n_workers (int): Number of workers for the dataloaders.
            with_pseudo_labelling (bool): Whether to use pseudo labelling.
            seed (Optional[int]): Seed for random number generation.
        """
        super(SkyFinderCoverModule, self).__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.with_pseudo_labelling = with_pseudo_labelling
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation and testing.

        Args:
            stage (Optional[str]): The stage of the training process. Can be "fit", "validate", "test" or None.
        """
        if self.seed is not None:
            print(f"🌱 Setting the seed to {self.seed} for generating dataloaders.")
            set_seed(self.seed)
        if self.with_pseudo_labelling:
            print("➕ Using pseudo labelling for training.")

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = SkyFinderCoverDataset(
                path=f"{SKY_FINDER_COVER_PATH}train",
                use_augmentations=True,
                with_pseudo_labelling=self.with_pseudo_labelling,
            )
            self.val_dataset = SkyFinderCoverDataset(
                path=f"{SKY_FINDER_COVER_PATH}val",
                use_augmentations=False,
                with_pseudo_labelling=self.with_pseudo_labelling,
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

        Returns:
            Optional[DataLoader]: The test dataloader, none by default.
        """
        return None
