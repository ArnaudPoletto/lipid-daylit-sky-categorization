import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss.
    """

    def __init__(self) -> None:
        """
        Initialize the Dice loss.

        Args:
            weight (torch.Tensor, optional): The weight, defaults to None
            size_average (bool, optional): Whether to average the loss, defaults to True
        """
        super(DiceLoss, self).__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): The inputs
            targets (torch.Tensor): The targets
            smooth (int, optional): The smoothing factor, defaults to 1

        Returns:
            torch.Tensor: The loss
        """
        inputs = F.sigmoid(inputs).type(torch.float32)
        targets = targets.type(torch.float32)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
