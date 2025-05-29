import os
import sys
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.contrastive_net import ContrastiveNet
from src.lightning_models.contrastive_lightning_model import ContrastiveLightningModel
from src.config import (
    CONTRASTIVE_CHECKPOINT_PATH,
    SKY_FINDER_HEIGHT,
    SKY_FINDER_WIDTH,
    PROJECTION_DIM,
    DEVICE,
)

def get_model() -> ContrastiveNet:
    """
    Get the texture descriptor model, based on the ContrastiveNet architecture.

    Raises:
        FileNotFoundError: If the contrastive checkpoint file does not exist.

    Returns:
        ContrastiveNet: An instance of the ContrastiveNet model.
    """
    if not os.path.exists(CONTRASTIVE_CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ Contrastive checkpoint not found at {CONTRASTIVE_CHECKPOINT_PATH}.")
    
    model = ContrastiveNet(
        projection_dim=PROJECTION_DIM, 
        pretrained=True,
        normalize_embeddings=True,
    )
    lightning_model = ContrastiveLightningModel.load_from_checkpoint(
        CONTRASTIVE_CHECKPOINT_PATH,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="contrastive_net",
        dataset="sky_finder",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()

    return model
    

def get_texture_descriptor(
    frame: np.ndarray,
    model: ContrastiveNet,
) -> np.ndarray:
    transform  = A.Compose(
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
    with torch.no_grad():
        features = model(frame)
    features = features.cpu().numpy()
    
    return features[0]
