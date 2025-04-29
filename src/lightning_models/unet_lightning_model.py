import os
import sys
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.losses.dice_loss import DiceLoss
from src.losses.focal_loss import FocalLoss


class UNetLightningModel(pl.LightningModule):
    """
    Lightning model for training and evaluating a UNet segmentation model.
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
        Initialize the UNetLightningModel.

        Args:
            model (nn.Module): The UNet model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            name (str): Name of the model.
            dataset (str): Name of the dataset.
        """
        super(UNetLightningModel, self).__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.name = name
        self.dataset = dataset
        self.cloud_criterion1 = FocalLoss(alpha=0.5, gamma=2.0)
        self.bloom_criterion1 = FocalLoss(alpha=0.9, gamma=1.0)
        self.cloud_criterion2 = DiceLoss()
        self.bloom_criterion2 = DiceLoss()

    def on_train_epoch_start(self) -> None:
        """
        Called at the start of the training epoch.
        """
        self.model.train()

    def on_validation_epoch_start(self) -> None:
        """
        Called at the start of the validation epoch.
        """
        self.model.eval()

    def on_test_epoch_start(self) -> None:
        """
        Called at the start of the test epoch.
        """
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        return self.model(x)

    def _shared_step(self, batch: List[torch.Tensor], step_type: str) -> torch.Tensor:
        """
        Shared step for training, validation, and testing.

        Args:
            batch (List[torch.Tensor]): Input batch containing [images, ground_truth].
            batch_idx (int): Index of the batch.
            step_type (str): Type of step ('train', 'val', or 'test').

        Returns:
            torch.Tensor: Loss value.
        """
        x, cloud_y, bloom_y = batch
        y_pred = self.model(x)
        if isinstance(y_pred, dict):
            y_pred = y_pred["out"]
        cloud_y_pred = y_pred[:, 0, :, :].unsqueeze(1)
        bloom_y_pred = y_pred[:, 1, :, :].unsqueeze(1)

        cloud_loss1 = self.cloud_criterion1(cloud_y_pred, cloud_y)
        cloud_loss2 = self.cloud_criterion2(cloud_y_pred, cloud_y)
        cloud_loss = cloud_loss1 + cloud_loss2
        bloom_loss1 = self.bloom_criterion1(bloom_y_pred, bloom_y)
        bloom_loss2 = self.bloom_criterion2(bloom_y_pred, bloom_y)
        bloom_loss = bloom_loss1 + bloom_loss2
        # loss = cloud_loss + bloom_loss
        loss = cloud_loss

        self.log(
            f"{step_type}_cloud_loss",
            cloud_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}_bloom_loss",
            bloom_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.

        Args:
            batch (List[torch.Tensor]): Input batch containing [images, ground_truth].
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step for the model.

        Args:
            batch (List[torch.Tensor]): Input batch containing [images, ground_truth].
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step for the model.

        Args:
            batch (List[torch.Tensor]): Input batch containing [images, ground_truth].
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        return self._shared_step(batch, "test")

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx=None
    ) -> torch.Tensor:
        """
        Prediction step for the model. This method is not used.
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
