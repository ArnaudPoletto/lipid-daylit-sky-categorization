import os
import sys
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.unet import UNet
from src.lightning_models.unet_lightning_model import UNetLightningModel
from src.config import (
    UNET_ACTIVE_CHECKPOINT_PATH,
    SKY_FINDER_HEIGHT,
    SKY_FINDER_WIDTH,
    DEVICE,
)

def get_model() -> UNet:
    """
    Load the pretrained unet model for sky classification.

    Returns:
        UNet: The pre-trained UNet model.
    """
    model = UNet(pretrained=True).to(DEVICE)
    lightning_model = UNetLightningModel.load_from_checkpoint(
        UNET_ACTIVE_CHECKPOINT_PATH,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()

    return model

def get_sky_cover(
        frame: np.ndarray,
        model: UNet,
) -> float:
    """
    Get the sky cover for a given frame using the UNet model.

    Args:
        frame (np.ndarray): The input image frame in BGR format.
        model (UNet): The pretrained UNet model.

    Returns:
        np.ndarray: The predicted sky cover mask.
    """
    # Preprocess the frame
    transform = A.Compose(
        [
            A.Resize(height=SKY_FINDER_HEIGHT, width=SKY_FINDER_WIDTH, p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(
                p=1.0,
            ),
        ]
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(image=frame)["image"]
    frame = frame.unsqueeze(0).to(DEVICE)

    # Get the sky cover
    with torch.no_grad():
        y_pred, _ = model(frame)
        y_pred = torch.sigmoid(y_pred[0, 0, :, :])  # Get the first channel
    y_pred = y_pred.cpu().numpy()

    sky_cover = np.mean(y_pred)

    return float(sky_cover)