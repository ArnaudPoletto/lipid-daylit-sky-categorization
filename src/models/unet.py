import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class UNet(nn.Module):
    """
    True UNet architecture with ResNet50 encoder backbone.
    """
    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        """
        Initialize the UNet module.

        Args:
            pretrained (bool): Whether to use pretrained weights for the encoder backbone.
        """
        super().__init__()

        # Encoder - using ResNet50 as backbone
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)

        # Extract layers from ResNet to use as encoder stages
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # 64 channels, 1/2 resolution
        self.pool1 = resnet.maxpool  # 1/4 resolution
        self.encoder2 = resnet.layer1  # 256 channels, 1/4 resolution
        self.encoder3 = resnet.layer2  # 512 channels, 1/8 resolution
        self.encoder4 = resnet.layer3  # 1024 channels, 1/16 resolution
        self.encoder5 = resnet.layer4  # 2048 channels, 1/32 resolution

        # Decoder path
        self.decoder5 = self._make_decoder_block(2048, 1024)  # 1/16 resolution
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)  # 1/8 resolution
        self.decoder3 = self._make_decoder_block(512 + 512, 256)  # 1/4 resolution
        self.decoder2 = self._make_decoder_block(256 + 256, 64)  # 1/2 resolution
        self.decoder1 = self._make_decoder_block(64 + 64, 32)  # full resolution

        # Final output layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with upsampling and convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Segmentation output of shape (batch_size, 1, height, width).
        """
        # Remember input size for later resizing
        input_size = x.size()[2:]
        
        # Encoder path
        e1 = self.encoder1(x)  # 1/2 resolution
        p1 = self.pool1(e1)    # 1/4 resolution
        e2 = self.encoder2(p1) # 1/4 resolution
        e3 = self.encoder3(e2) # 1/8 resolution
        e4 = self.encoder4(e3) # 1/16 resolution
        e5 = self.encoder5(e4) # 1/32 resolution

        # Decoder path with skip connections
        # Upsample e5 to match e4 dimensions
        up5 = F.interpolate(e5, size=e4.size()[2:], mode='bilinear', align_corners=False)
        d5 = self.decoder5(up5)
        
        # Upsample d5 to match e3 dimensions
        up4 = F.interpolate(torch.cat([d5, e4], dim=1), size=e3.size()[2:], mode='bilinear', align_corners=False)
        d4 = self.decoder4(up4)
        
        # Upsample d4 to match e2 dimensions
        up3 = F.interpolate(torch.cat([d4, e3], dim=1), size=e2.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(up3)
        
        # Upsample d3 to match e1 dimensions
        up2 = F.interpolate(torch.cat([d3, e2], dim=1), size=e1.size()[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(up2)
        
        # Upsample d2 to match original input dimensions (or next layer if needed)
        up1 = F.interpolate(torch.cat([d2, e1], dim=1), scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.decoder1(up1)
        
        # Final output layer and resize to input resolution
        output = self.final_conv(d1)
        if output.size()[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return self.sigmoid(output)