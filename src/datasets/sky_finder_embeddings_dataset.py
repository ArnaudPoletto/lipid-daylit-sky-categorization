import os
import sys
import json
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.random import set_seed
from src.utils.random import SeededDataLoader
from src.config import (
    SKY_FINDER_DESCRIPTORS_PATH,
)

class SkyFinderEmbeddingsDataset(Dataset):
    """
    Sky Finder embeddings dataset.
    """

    def _load_data(self) -> List[Tuple[Dict[str, Any], str]]:
        # Load the embeddings data from the JSON file.
        if not os.path.exists(SKY_FINDER_DESCRIPTORS_PATH):
            raise FileNotFoundError(f"❌ Sky Finder descriptors file not found at {os.path.abspath(SKY_FINDER_DESCRIPTORS_PATH)}.")
        
        with open(SKY_FINDER_DESCRIPTORS_PATH, 'r') as file:
            data = json.load(file)
            data = data.get(self.dataset_split, [])

        if not data:
            raise ValueError(f"❌ No data found for the split '{self.dataset_split}' in the embeddings file.")
        
        # Convert the data into a list of tuples (descriptor, label).
        processed_data = []
        for sky_class, camera_dict in data.items():
            for camera_id, samples in camera_dict.items():
                for sample_name, descriptors in samples.items():
                    processed_data.append((descriptors["sky_image_descriptor"], sky_class))

        return processed_data


    def __init__(
        self,
        dataset_split: str,
    ) -> None:
        """
        Initialize the Sky Finder embeddings dataset.

        Args:
            dataset_split (str): The split of the dataset to use (e.g., 'train', 'val', 'test').
        """
        super(SkyFinderEmbeddingsDataset, self).__init__()

        if dataset_split not in ['train', 'val', 'test']:
            raise ValueError(f"❌ Invalid dataset split: {dataset_split}. Must be one of 'train', 'val', or 'test'.")
        
        self.dataset_split = dataset_split
        self.data = self._load_data()

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            torch.Tensor: The processed image patch pair.
        """
        sample = self.data[idx]
        sky_image_descriptor, sky_class = sample

        sky_image_descriptor = np.array(sky_image_descriptor, dtype=np.float32)
        sky_image_descriptor = torch.tensor(sky_image_descriptor, dtype=torch.float32)
        
        label = np.array(
            [
                1.0 if sky_class == "clear" else 0.0,
                1.0 if sky_class == "partial" else 0.0,
                1.0 if sky_class == "overcast" else 0.0,
            ], dtype=np.float32
        )
        label = torch.tensor(label, dtype=torch.float32)

        return sky_image_descriptor, label
    

class SkyFinderEmbeddingsModule(pl.LightningDataModule):
    """
    Sky Finder embeddings data module.
    """

    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Sky Finder embeddings data module.

        Args:
            batch_size (int): The batch size for the dataloaders.
            n_workers (int): The number of workers for the dataloaders.
            seed (Optional[int]): The seed for random number generation.
        """
        super(SkyFinderEmbeddingsModule, self).__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
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

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = SkyFinderEmbeddingsDataset(
                dataset_split="train",
            )
            self.val_dataset = SkyFinderEmbeddingsDataset(
                dataset_split="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = SkyFinderEmbeddingsDataset(
                dataset_split="test",
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