import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Implementation of the Dice Loss for segmentation tasks.
    Supports both binary and continuous (0 to 1) target values.
    """

    def __init__(self, smooth=1.0, square=True) -> None:
        """
        Initialize the DiceLoss module.

        Args:
            smooth (float): Small constant added to numerator and denominator to avoid division by zero.
            square (bool): Whether to square the inputs before calculating the loss.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Dice Loss.

        Args:
            predictions (torch.Tensor): Predicted probability masks, shape (batch_size, 1, height, width).
            targets (torch.Tensor): Ground truth masks, shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        # Flatten the tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Apply squaring if enabled
        if self.square:
            intersection = torch.sum(predictions * targets)
            pred_sum = torch.sum(predictions * predictions)
            target_sum = torch.sum(targets * targets)
        else:
            intersection = torch.sum(predictions * targets)
            pred_sum = torch.sum(predictions)
            target_sum = torch.sum(targets)

        # Calculate Dice coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Convert to loss (1 - dice)
        dice_loss = 1.0 - dice_coefficient

        return dice_loss