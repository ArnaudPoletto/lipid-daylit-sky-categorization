import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any
import lightning.pytorch as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.losses.ntxent_loss import NTXentLoss
from src.config import (
    CRITERION_TEMPERATURE,
)


class LightningModel(pl.LightningModule):
    """
    Lightning model for training and evaluating a neural network model.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        name: str,
        dataset: str,
    ) -> None:
        """
        Initialize the LightningModel.

        Args:
            model (nn.Module): The neural network model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            name (str): Name of the model.
            dataset (str): Name of the dataset.
        """
        super(LightningModel, self).__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.name = name
        self.dataset = dataset
        self.criterion = NTXentLoss(temperature=CRITERION_TEMPERATURE)

    def forward(self, batch: torch.Tensor) -> None:
        """
        Forward pass through the model. This method is not used.
        """
        pass

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.

        Args:
            batch (Any): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        # Resize batch and forward pass
        batch_size, n_pairs, two, n_channels, height, width = batch.shape
        flattened_batch = batch.view(
            batch_size * n_pairs * two, n_channels, height, width
        )
        flattened_outputs = self.model(flattened_batch)
        outputs = flattened_outputs.view(batch_size, n_pairs, two, -1)

        # Compute and log loss
        train_loss = self.criterion(outputs)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the model.

        Args:
            batch (Any): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        # Resize batch and forward pass
        batch_size, n_pairs, two, n_channels, height, width = batch.shape
        flattened_batch = batch.view(
            batch_size * n_pairs * two, n_channels, height, width
        )
        flattened_outputs = self.model(flattened_batch)
        outputs = flattened_outputs.view(batch_size, n_pairs, two, -1)

        # Compute and log loss
        val_loss = self.criterion(outputs)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)

        return val_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Test step for the model.

        Args:
            batch (Any): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        # Resize batch and forward pass
        batch_size, n_pairs, two, n_channels, height, width = batch.shape
        flattened_batch = batch.view(
            batch_size * n_pairs * two, n_channels, height, width
        )
        flattened_outputs = self.model(flattened_batch)
        outputs = flattened_outputs.view(batch_size, n_pairs, two, -1)

        # Compute and log loss
        test_loss = self.criterion(outputs)
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, sync_dist=True)

        return test_loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx=None
    ) -> None:
        """
        Prediction step for the model. This method is not used.

        Args:
            batch (Any): Input batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to None.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler for the model.

        Returns:
            Dict[str, Any]: Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": learning_rate_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }
