import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.sky_class_net import SkyClassNet
from src.lightning_models.sky_class_lightning_model import SkyClassLightningModel
from src.config import (
    SKY_CLASS_NET_CHECKPOINT_PATH,
    DEVICE,
)


def get_model() -> SkyClassNet:
    """
    Load the pretrained SkyClassNet model for sky classification.
    
    This function loads a neural network model that classifies sky conditions into
    three categories based on sky image descriptors. The model takes 16-dimensional
    feature vectors as input and outputs probabilities for each sky class.

    Returns:
        SkyClassNet: The pre-trained sky classification model configured for inference.
        
    Raises:
        FileNotFoundError: If the checkpoint file does not exist at SKY_CLASS_NET_CHECKPOINT_PATH.
    """
    # Initialize SkyClassNet architecture
    model = SkyClassNet(
        input_dim=16,
        output_dim=3,
        dropout_rate=0.0,
    ).to(DEVICE)
    
    # Load trained weights from Lightning checkpoint
    lightning_model = SkyClassLightningModel.load_from_checkpoint(
        SKY_CLASS_NET_CHECKPOINT_PATH,
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


def get_sky_class(
    sky_image_descriptor: np.ndarray,
    model: SkyClassNet,
) -> np.ndarray:
    """
    Get the sky class for a given sky image descriptor using the SkyClassNet model.
    
    This function classifies sky conditions into one of three categories:
    - Class 0: Clear sky
    - Class 1: Partially cloudy sky
    - Class 2: Overcast sky
    
    The classification is based on a 16-dimensional feature vector extracted from
    sky images using a contrastive learning model.

    Args:
        sky_image_descriptor (np.ndarray): The unnormalized sky image descriptor with shape (16,) containing feature values.
        model (SkyClassNet): The pre-trained SkyClassNet model.
        
    Returns:
        int: The predicted sky class index (0, 1, or 2).
    """
    x = torch.tensor(sky_image_descriptor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    y_pred = model(x)
    y_pred = torch.softmax(y_pred, dim=1)
    sky_class = torch.argmax(y_pred, dim=1).item()

    return sky_class