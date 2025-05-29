import torch
import torch.nn as nn

class SkyClassNet(nn.Module):
    """
    Classification head for sky classification.
    """
    def __init__(
        self,
        input_dim: int = 17,
        output_dim: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the SkyClassNet module.
        """
        super(SkyClassNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 8)),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 8), max(input_dim // 4, 4)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 4, 4), output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)

