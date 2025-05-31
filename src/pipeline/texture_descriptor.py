import os
import sys
import cv2
import json
import umap
import torch
import matplotlib
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from adjustText import adjust_text
from typing import Optional, List, Tuple
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
        normalize_embeddings=False,
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

    # Get the texture descriptor
    with torch.no_grad():
        features = model(frame)
    features = features.cpu().numpy()

    return features[0]


def get_sky_finder_texture_descriptors() -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(SKY_FINDER_DESCRIPTORS_PATH):
        raise FileNotFoundError(
            f"❌ Sky Finder descriptors not found at {SKY_FINDER_DESCRIPTORS_PATH}. Please generate them first by running the script at src/classification/generate_sky_finder_descriptors.py."
        )

    # Load the Sky Finder texture descriptors from the JSON file
    with open(SKY_FINDER_DESCRIPTORS_PATH, "r") as f:
        sky_finder_descriptors = json.load(f)
    test_sky_finder_descriptors = sky_finder_descriptors["test"]

    # Get the texture descriptors for the test set
    test_sky_finder_texture_descriptors_dict = {}
    sky_classes = []
    for sky_class, camera_dict in test_sky_finder_descriptors.items():
        for camera_id, sample_dict in camera_dict.items():
            for sample_id, descriptors in sample_dict.items():
                test_sky_finder_texture_descriptors_dict[
                    f"{sky_class}_{camera_id}_{sample_id}"
                ] = np.array(descriptors["contrastive_embeddings"])
                sky_classes.append(sky_class)

    # Ensure the descriptors are in the correct format
    n_samples = len(test_sky_finder_texture_descriptors_dict)
    sky_finder_texture_descriptors = np.zeros(
        (n_samples, PROJECTION_DIM), dtype=np.float32
    )
    for i, (key, value) in enumerate(test_sky_finder_texture_descriptors_dict.items()):
        sky_finder_texture_descriptors[i] = value

    return sky_finder_texture_descriptors, sky_classes

def get_fitted_umap_reducer(
    sky_finder_texture_descriptors: np.ndarray,
    n_neighbors: int = 100,
    min_dist: float = 0.1,
) -> umap.UMAP:
    print("⏳ Fitting UMAP reducer to Sky Finder texture descriptors...")
    # Get UMAP and fit data
    reducer = umap.UMAP(
        metric="cosine",
        n_components=2,
        random_state=SEED,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    reducer.fit(sky_finder_texture_descriptors)
    print("✅ UMAP reducer fitted successfully.")

    return reducer

def plot_sky_finder_texture_descriptors(
    fitted_umap: umap.UMAP,
    sky_finder_texture_descriptors: np.ndarray,
    colors: Optional[List[str]] = None,
    oos_texture_descriptors: Optional[np.ndarray] = None,
    oos_colors: Optional[List[str]] = None,
    oos_labels: Optional[List[str]] = None,
) -> None:
    if oos_texture_descriptors is not None and oos_labels is not None and len(oos_texture_descriptors) != len(oos_labels):
        raise ValueError(
            "❌ The number of out-of-sample texture descriptors must match the number of labels."
        )
    if colors is not None and len(colors) != len(sky_finder_texture_descriptors):
        raise ValueError(
            "❌ The number of colors must match the number of Sky Finder texture descriptors."
        )
    
    projected_descriptors = fitted_umap.transform(sky_finder_texture_descriptors)
    if oos_texture_descriptors is not None:
        projected_oos_descriptors = fitted_umap.transform(oos_texture_descriptors)
    else:
        projected_oos_descriptors = None

    plt.figure(figsize=(15, 15))
    plt.scatter(
        projected_descriptors[:, 0],
        projected_descriptors[:, 1],
        s=10,
        alpha=0.7 if oos_texture_descriptors is None else 0.1,
        color=colors if colors is not None else "blue",
    )
    if projected_oos_descriptors is not None:
        scatter_points = plt.scatter(
            projected_oos_descriptors[:, 0],
            projected_oos_descriptors[:, 1],
            s=50,
            alpha=1.0,
            color=oos_colors if oos_colors is not None else "green",
            marker="X",
        )

        # Add labels for out-of-sample descriptors
        if oos_labels is not None:
            texts = []
            for i, label in enumerate(oos_labels):
                text = plt.text(
                    projected_oos_descriptors[i, 0],
                    projected_oos_descriptors[i, 1],
                    label,
                    fontsize=9,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(text)

            adjust_text(
                texts,
                x=projected_oos_descriptors[:, 0],
                y=projected_oos_descriptors[:, 1],
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.5),
                expand_points=(2.5, 2.5),
                expand_text=(1.5, 1.5),
                expand_objects=(0.5, 0.5),
                force_points=(0.1, 0.1),
                force_text=(0.8, 0.9),
                force_objects=(0.8, 0.8),
                objects=scatter_points,
            )

    plt.title("UMAP Visualization of Sky Finder Test Set Texture Descriptors")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()