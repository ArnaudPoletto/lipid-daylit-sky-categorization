import os
import sys
import cv2
import torch
import numpy as np
import albumentations as A
import lightning.pytorch as pl
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.random import set_seed
from src.utils.random import SeededDataLoader
from src.utils.sky_finder import (
    get_sky_finder_masks,
    get_sky_finder_bounding_boxes,
    get_splitted_sky_finder_paths_dict,
)
from src.config import (
    SKY_FINDER_WIDTH,
    SKY_FINDER_HEIGHT,
)

class SkyFinderDataset(Dataset):
    """
    Sky Finder dataset.
    """

    def _create_indices(self) -> List[str]:
        """
        Create a flat list of image paths from the nested paths dictionary.
        
        Returns:
            List[str]: A flat list of image paths.
        """
        flat_path_list = []
        for sky_class, camera_dict in self.paths_dict.items():
            for camera_id, image_paths in camera_dict.items():
                for image_path in image_paths:
                    flat_path_list.append(
                        (camera_id, sky_class, image_path)
                    )

        return flat_path_list

    def __init__(
        self,
        paths_dict: Dict[str, Dict[str, List[str]]],
    ) -> None:
        """
        Initialize the Sky Finder dataset.

        Args:
            paths_dict (Dict[str, Dict[str, List[str]]]): A dictionary containing paths to images.
                The keys are sky classes, and the values are dictionaries with camera IDs as keys
                and lists of image paths as values.
        """
        super(SkyFinderDataset, self).__init__()

        self.paths_dict = paths_dict

        self.all_images = self._create_indices()
        self.masks = get_sky_finder_masks(paths_dict)
        self.bounding_boxes = get_sky_finder_bounding_boxes(paths_dict)

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

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            torch.Tensor: The processed image patch pair.
        """
        camera_id, sky_class, image_path = self.all_images[idx]
        # Load the image
        image = self._get_image(image_path)
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

        # Resize and normalize the image
        transform = A.Compose(
            [
                A.Resize(SKY_FINDER_HEIGHT, SKY_FINDER_WIDTH, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(p=1.0),
            ]
        )
        processed_image = transform(image=filled_image)["image"]
        image_name = image_path.split("/")[-1]

        dataset_split = image_path.split("/")[-4]

        return dataset_split, sky_class, camera_id, image_name, mask, processed_image
    

class SkyFinderModule(pl.LightningDataModule):
    """
    Sky Finder data module.
    """

    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Sky Finder data module.

        Args:
            batch_size (int): The batch size for the dataloaders.
            n_workers (int): The number of workers for the dataloaders.
            seed (Optional[int]): The seed for random number generation.
        """
        super(SkyFinderModule, self).__init__()

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
            self.train_dataset = SkyFinderDataset(
                paths_dict=self.train_paths_dict,
            )
            self.val_dataset = SkyFinderDataset(
                paths_dict=self.val_paths_dict,
            )
        if stage == "test" or stage is None:
            self.test_dataset = SkyFinderDataset(
                paths_dict=self.test_paths_dict,
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