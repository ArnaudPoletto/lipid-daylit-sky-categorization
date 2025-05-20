import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
    ) -> None:
        """
        Initialize the Focal loss.

        Args:
            alpha (float): Weighting factor for the rare class.
            gamma (float): Focusing parameter that controls down-weighting of easy examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): The logits from the model (pre-sigmoid)
            targets (torch.Tensor): The target values (0, 0.5, 1)

        Returns:
            torch.Tensor: The loss
        """
        # Apply sigmoid to get probabilities
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=self.eps, max=1.0 - self.eps)

        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        # Calculate the focal weight
        if targets.dtype != torch.float32:
            targets = targets.type(torch.float32)

        # Calculate p_t - probability of the target class
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            # Apply alpha weight to focal weight
            # Alpha for target=1, (1-alpha) for target=0
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_weight * focal_weight

        # Calculate focal loss
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()
