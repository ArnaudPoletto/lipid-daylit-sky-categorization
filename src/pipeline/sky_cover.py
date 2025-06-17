import os
import sys
import cv2
import torch
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
    Load the pretrained UNet model for sky cover prediction.
    
    This function loads a UNet model that has been trained to predict cloud coverage
    in sky regions. The model outputs a segmentation mask indicating cloudy areas.

    Returns:
        UNet: The pre-trained UNet model configured for inference.
        
    Raises:
        FileNotFoundError: If the checkpoint file does not exist at UNET_ACTIVE_CHECKPOINT_PATH.
    """
    # Initialize UNet architecture with pretrained weights
    model = UNet(pretrained=True).to(DEVICE)
    
    # Load trained weights from Lightning checkpoint
    lightning_model = UNetLightningModel.load_from_checkpoint(
        UNET_ACTIVE_CHECKPOINT_PATH,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    
    # Extract the core model and set to evaluation mode
    model = lightning_model.model.to(DEVICE)
    model.eval()

    return model


def get_sky_cover(
    frame: np.ndarray,
    model: UNet,
) -> float:
    """
    Get the sky cover percentage for a given frame using the UNet model.
    
    This function processes an input frame through a UNet model to predict cloud coverage.
    The model outputs a probability mask where higher values indicate the presence of clouds.
    The final sky cover value is computed as the mean of all predicted probabilities.

    Args:
        frame (np.ndarray): The input image frame in BGR format with shape (H, W, 3).
        model (UNet): The pretrained UNet model for cloud segmentation.

    Returns:
        float: The predicted sky cover as a percentage value between 0.0 and 1.0,
            where 0.0 indicates clear sky and 1.0 indicates fully cloudy sky.
    """
    # Configure preprocessing pipeline
    transform = A.Compose(
        [
            A.Resize(height=SKY_FINDER_HEIGHT, width=SKY_FINDER_WIDTH, p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )
    
    # Convert BGR to RGB and apply transformations
    frame = transform(image=frame)["image"]
    frame = frame.unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        y_pred, _ = model(frame)
        y_pred = torch.sigmoid(y_pred[0, 0, :, :])
    
    # Convert to numpy and compute mean cloud coverage
    y_pred = y_pred.cpu().numpy()
    sky_cover = np.mean(y_pred)

    return float(sky_cover)