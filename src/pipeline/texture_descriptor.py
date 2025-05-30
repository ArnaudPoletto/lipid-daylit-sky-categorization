import os
import sys
import cv2
import json
import torch
import matplotlib
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from albumentations.pytorch.transforms import ToTensorV2

matplotlib.use("TkAgg")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.contrastive_net import ContrastiveNet
from src.lightning_models.contrastive_lightning_model import ContrastiveLightningModel
from src.config import (
    SKY_FINDER_DESCRIPTORS_PATH,
    CONTRASTIVE_CHECKPOINT_PATH,
    SKY_FINDER_HEIGHT,
    SKY_FINDER_WIDTH,
    PROJECTION_DIM,
    DEVICE,
    SEED,
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
        raise FileNotFoundError(
            f"❌ Contrastive checkpoint not found at {CONTRASTIVE_CHECKPOINT_PATH}."
        )

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
    """
    Get the texture descriptor for a given frame using the ContrastiveNet model.

    Args:
        frame (np.ndarray): The input image frame in BGR format.
        model (ContrastiveNet): The pre-trained ContrastiveNet model.

    Returns:
        np.ndarray: The texture descriptor for the input frame.
    """
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
    with torch.no_grad():
        features = model(frame)
    features = features.cpu().numpy()

    return features[0]


def get_sky_finder_texture_descriptors() -> np.ndarray:
    if not os.path.exists(SKY_FINDER_DESCRIPTORS_PATH):
        raise FileNotFoundError(
            f"❌ Sky Finder descriptors not found at {SKY_FINDER_DESCRIPTORS_PATH}. Please generate them first by running the script at src/classification/generate_sky_finder_descriptors.py."
        )

    with open(SKY_FINDER_DESCRIPTORS_PATH, "r") as f:
        sky_finder_descriptors = json.load(f)
    test_sky_finder_descriptors = sky_finder_descriptors["test"]
    test_sky_finder_texture_descriptors_dict = {}
    for sky_class, camera_dict in test_sky_finder_descriptors.items():
        for camera_id, sample_dict in camera_dict.items():
            for sample_id, descriptors in sample_dict.items():
                test_sky_finder_texture_descriptors_dict[
                    f"{sky_class}_{camera_id}_{sample_id}"
                ] = np.array(descriptors["contrastive_embeddings"])

    n_samples = len(test_sky_finder_texture_descriptors_dict)
    sky_finder_texture_descriptors = np.zeros(
        (n_samples, PROJECTION_DIM), dtype=np.float32
    )
    for i, (key, value) in enumerate(test_sky_finder_texture_descriptors_dict.items()):
        sky_finder_texture_descriptors[i] = value

    return sky_finder_texture_descriptors


def plot_sky_finder_texture_descriptors(
    sky_finder_texture_descriptors: np.ndarray,
    perplexity: int = 100,
) -> TSNE:
    # Get TSNE and fit data
    print("=>", sky_finder_texture_descriptors.shape)
    tsne = TSNE(
        metric="cosine",
        n_components=2,
        random_state=SEED,
        perplexity=perplexity,
    )
    projected_descriptors = tsne.fit_transform(sky_finder_texture_descriptors)

    plt.figure(figsize=(10, 10))
    plt.scatter(
        projected_descriptors[:, 0],
        projected_descriptors[:, 1],
        s=10,
        alpha=0.7,
    )
    plt.title("T-SNE Visualization of Sky Finder Test Set Texture Descriptors")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.show()

    return tsne