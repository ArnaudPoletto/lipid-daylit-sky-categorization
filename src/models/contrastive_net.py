import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class ContrastiveNet(nn.Module):
    def __init__(
        self, 
        projection_dim: int = 256, 
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
        
        hidden_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projection = self.projector(features)
        normalized_projection = F.normalize(projection, p=2, dim=1)
        
        return normalized_projection