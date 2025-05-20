import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class ContrastiveNet(nn.Module):
    """
    Contrastive Neural Network using ResNet50 as backbone.
    """

    def __init__(
        self,
        projection_dim: int,
        pretrained: bool,
    ) -> None:
        """
        Initialize the ContrastiveNet module.

        Args:
            projection_dim (int): Dimension of the projection head.
            pretrained (bool): Whether to use pretrained weights for the ResNet50 backbone.
        """
        super().__init__()

        # Use ResNet50 as backbone encoder
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

            # Change classification head by a new projection head
        hidden_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, projection_dim).
        """
        features = self.backbone(x)
        projection = self.projector(features)
        normalized_projection = F.normalize(projection, p=2, dim=1)

        return normalized_projection
