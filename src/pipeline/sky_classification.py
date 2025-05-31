import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.sky_class_net import SkyClassNet
from src.lightning_models.sky_class_lightning_model import SkyClassLightningModel
from src.config import (
    CONTRASTIVE_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH,
    DEVICE,
)

def get_model():
    model = SkyClassNet(
        input_dim=16,
        output_dim=3,
        dropout_rate=0.0,
    ).to(DEVICE)
    lightning_model = SkyClassLightningModel.load_from_checkpoint(
        CONTRASTIVE_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()

    return model

def get_sky_class(
    texture_descriptor: np.ndarray,
    model: SkyClassNet,
) -> np.ndarray:
    """
    Get the sky class for a given frame using the SkyClassNet model.
    Args:
        texture_descriptor (np.ndarray): The unnormalized texture descriptor for the frame.
        model (SkyClassNet): The pre-trained SkyClassNet model.
        
    Returns:
        str: The predicted sky class label ("clear", "partial", "overcast").
    """
    x = torch.tensor(texture_descriptor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    y_pred = model(x)
    y_pred = torch.softmax(y_pred, dim=1)
    sky_class = torch.argmax(y_pred, dim=1).item()

    return sky_class