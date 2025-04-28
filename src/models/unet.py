import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class UNet(nn.Module):
    """
    UNet architecture with ResNet50 encoder backbone.
    """
    def __init__(
        self,
        pretrained: bool = True,
        bottleneck_dropout_rate: float = 0.4,
        decoder_dropout_rate: float = 0.4,
    ) -> None:
        """
        Initialize the UNet module.

        Args:
            pretrained (bool): Whether to use pretrained weights for the encoder backbone.
            bottleneck_dropout_rate (float): Dropout rate for the bottleneck layer.
            decoder_dropout_rate (float): Dropout rate for the decoder layers.
        """
        super().__init__()

        self.bottleneck_dropout_rate = bottleneck_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.bottleneck_dropout = nn.Dropout2d(bottleneck_dropout_rate)

        # Use ResNet50 as backbone encoder
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)

        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder path
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 32)

        # Final output layer
        self.final_conv = nn.Conv2d(32, 2, kernel_size=1)

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a decoder block with upsampling and convolutions.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        
        Returns:
            nn.Sequential: A sequential block containing upsampling and convolutions.
        """
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(self.decoder_dropout_rate),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(self.decoder_dropout_rate),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Segmentation output of shape (batch_size, 1, height, width).
        """
        input_size = x.size()[2:]
        
        # Encoder path
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        if self.bottleneck_dropout_rate > 0:
            e5 = self.bottleneck_dropout(e5)

        # Decoder path with skip connections
        up5 = F.interpolate(e5, size=e4.size()[2:], mode='bilinear', align_corners=False)
        d5 = self.decoder5(up5)
        
        up4 = F.interpolate(torch.cat([d5, e4], dim=1), size=e3.size()[2:], mode='bilinear', align_corners=False)
        d4 = self.decoder4(up4)
        
        up3 = F.interpolate(torch.cat([d4, e3], dim=1), size=e2.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(up3)
        
        up2 = F.interpolate(torch.cat([d3, e2], dim=1), size=e1.size()[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(up2)
        
        up1 = F.interpolate(torch.cat([d2, e1], dim=1), scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.decoder1(up1)
        
        # Final output layer and resize to input resolution
        output = self.final_conv(d1)
        if output.size()[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        return output