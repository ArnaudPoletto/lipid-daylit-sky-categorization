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
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from adjustText import adjust_text
from scipy.spatial import ConvexHull
from sklearn.preprocessing import normalize
from typing import Optional, List, Tuple, Dict
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
    SKY_FINDER_PATH,
    PROJECTION_DIM,
    DEVICE,
    SEED,
)


def get_model() -> ContrastiveNet:
    """
    Get the sky image descriptor model, based on the ContrastiveNet architecture.

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


def get_sky_image_descriptor(
    frame: np.ndarray,
    model: ContrastiveNet,
) -> np.ndarray:
    """
    Get the sky image descriptor for a given frame using the ContrastiveNet model.

    Args:
        frame (np.ndarray): The input image frame in BGR format.
        model (ContrastiveNet): The pretrained ContrastiveNet model.

    Returns:
        np.ndarray: The sky image descriptor for the input frame.
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
    frame = transform(image=frame)["image"]
    frame = frame.unsqueeze(0).to(DEVICE)

    # Get the sky image descriptor
    with torch.no_grad():
        features = model(frame)
    features = features.cpu().numpy()

    return features[0]

def get_kmeans_groups(
        sky_image_descriptors: np.ndarray,
        k: int,
) -> List[int]:
    """
    Group sky image descriptors using k-means clustering.

    Args:
        sky_image_descriptors (np.ndarray): The sky image descriptors to cluster.
        k (int): The number of clusters to form.

    Returns:
        List[int]: The cluster labels for each sky image descriptor.
    """
    normalized_descriptors = normalize(sky_image_descriptors, norm='l2')

    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_descriptors)

    return cluster_labels.tolist()



def get_sky_finder_descriptors() -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load Sky Finder sky image descriptors from the generated descriptors file.

    Raises:
        FileNotFoundError: If the Sky Finder descriptors file does not exist.

    Returns:
        np.ndarray: The sky image descriptors for the Sky Finder test set.
        List[float]: The cloud coverages for the Sky Finder test set.
        List[str]: The sky classes for the Sky Finder test set.
        List[str]: The image paths for the Sky Finder test set.
    """
    if not os.path.exists(SKY_FINDER_DESCRIPTORS_PATH):
        raise FileNotFoundError(
            f"❌ Sky Finder descriptors not found at {SKY_FINDER_DESCRIPTORS_PATH}. Please generate them first by running the script at src/classification/generate_sky_finder_descriptors.py."
        )

    # Load the Sky Finder sky image descriptors from the JSON file
    with open(SKY_FINDER_DESCRIPTORS_PATH, "r") as f:
        sky_finder_descriptors = json.load(f)
    test_sky_finder_descriptors = sky_finder_descriptors["test"]

    # Get the sky image descriptors for the test set
    test_sky_finder_sky_image_descriptors_dict = {}
    cloud_coverages = []
    sky_classes = []
    image_paths = []
    for sky_class, camera_dict in test_sky_finder_descriptors.items():
        for camera_id, sample_dict in camera_dict.items():
            for sample_id, descriptors in sample_dict.items():
                test_sky_finder_sky_image_descriptors_dict[
                    f"{sky_class}_{camera_id}_{sample_id}"
                ] = np.array(descriptors["sky_image_descriptor"])
                cloud_coverages.append(descriptors["cloud_coverage"])
                sky_classes.append(sky_class)
                image_path = f"{SKY_FINDER_PATH}test/{sky_class}/{camera_id}/{sample_id}"
                image_paths.append(image_path)

    # Ensure the descriptors are in the correct format
    n_samples = len(test_sky_finder_sky_image_descriptors_dict)
    sky_finder_sky_image_descriptors = np.zeros(
        (n_samples, PROJECTION_DIM), dtype=np.float32
    )
    for i, (key, value) in enumerate(test_sky_finder_sky_image_descriptors_dict.items()):
        sky_finder_sky_image_descriptors[i] = value

    return sky_finder_sky_image_descriptors, cloud_coverages, sky_classes, image_paths

def get_fitted_umap_reducer(
    sky_finder_sky_image_descriptors: np.ndarray,
    n_neighbors: int = 100,
    min_dist: float = 0.1,
) -> umap.UMAP:
    """
    Fit a UMAP reducer to Sky Finder sky image descriptors.

    Args:
        sky_finder_sky_image_descriptors (np.ndarray): The sky image descriptors to fit UMAP on.
        n_neighbors (int): The number of neighboring points used in local approximations of manifold structure.
        min_dist (float): The effective minimum distance between embedded points.

    Returns:
        umap.UMAP: The fitted UMAP reducer.
    """
    print("⏳ Fitting UMAP reducer to Sky Finder sky image descriptors...")
    # Get UMAP and fit data
    reducer = umap.UMAP(
        metric="cosine",
        n_components=2,
        random_state=SEED,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    reducer.fit(sky_finder_sky_image_descriptors)
    print("✅ UMAP reducer fitted successfully.")

    return reducer

def plot_sky_finder_sky_image_descriptors(
    fitted_umap: umap.UMAP,
    sky_finder_sky_image_descriptors: np.ndarray,
    colors: Optional[List[str]] = None,
    color_labels: Optional[Dict[str, str]] = None,
    image_paths: Optional[List[str]] = None,
    oos_sky_image_descriptors: Optional[np.ndarray] = None,
    oos_colors: Optional[List[str]] = None,
    oos_labels: Optional[List[str]] = None,
    oos_as_convex_hull: bool = False,
) -> None:
    """
    Plot Sky Finder sky image descriptors in 2D UMAP space.

    Args:
        fitted_umap (umap.UMAP): The fitted UMAP reducer.
        sky_finder_sky_image_descriptors (np.ndarray): The sky image descriptors to plot.
        colors (Optional[List[str]]): Colors for each descriptor point.
        color_labels (Optional[Dict[str, str]]): Mapping of colors to labels for legend.
        image_paths (Optional[List[str]]): Paths to images for interactive display.
        oos_sky_image_descriptors (Optional[np.ndarray]): Out-of-sample sky image descriptors.
        oos_colors (Optional[List[str]]): Colors for out-of-sample descriptors.
        oos_labels (Optional[List[str]]): Labels for out-of-sample descriptors.
        oos_as_convex_hull (bool): Whether to plot out-of-sample descriptors as convex hull.

    Raises:
        ValueError: If array dimensions don't match or invalid parameters.
    """
    if not oos_as_convex_hull and oos_sky_image_descriptors is not None and oos_labels is not None and len(oos_sky_image_descriptors) != len(oos_labels):
        raise ValueError(
            "❌ The number of out-of-sample sky image descriptors must match the number of labels."
        )
    if colors is not None and len(colors) != len(sky_finder_sky_image_descriptors):
        raise ValueError(
            "❌ The number of colors must match the number of Sky Finder sky image descriptors."
        )
    if image_paths is not None and len(image_paths) != len(sky_finder_sky_image_descriptors):
            raise ValueError("❌ The number of image paths must match the number of sky image descriptors.")
    if oos_as_convex_hull and oos_labels is not None and len(oos_labels) != 1:
        raise ValueError(
            "❌ If out-of-sample descriptors are plotted as a convex hull, there must be exactly one label."
        )
    
    
    # Project descriptors to 2D
    projected_descriptors = fitted_umap.transform(sky_finder_sky_image_descriptors)
    if oos_sky_image_descriptors is not None:
        projected_oos_descriptors = fitted_umap.transform(oos_sky_image_descriptors)
    else:
        projected_oos_descriptors = None

    # Create figure
    interactive = image_paths is not None
    if interactive:
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_main = fig.add_subplot(gs[0, 0])
        ax_image = fig.add_subplot(gs[0, 1])
        ax_image.set_title("Hover over points to see images")
        ax_image.axis('off')
    else:
        fig, ax_main = plt.subplots(figsize=(15, 15))

    # Plot main scatter
    ax_main.scatter(
        projected_descriptors[:, 0],
        projected_descriptors[:, 1],
        s=10,
        alpha=0.7 if oos_sky_image_descriptors is None else 0.1,
        color=colors if colors is not None else "blue",
    )

    # Plot out-of-sample descriptors if provided
    if projected_oos_descriptors is not None:
        if oos_as_convex_hull and len(projected_oos_descriptors) >= 3:
            hull = ConvexHull(projected_oos_descriptors)

            # Plot the convex hull area
            for simplex in hull.simplices:
                ax_main.plot(
                    projected_oos_descriptors[simplex, 0], 
                    projected_oos_descriptors[simplex, 1], 
                    color="gray", 
                    alpha=0.8,
                    linewidth=1
                )
            
            # Fill the convex hull area
            hull_points = projected_oos_descriptors[hull.vertices]
            ax_main.fill(
                hull_points[:, 0], 
                hull_points[:, 1], 
                color="gray", 
                alpha=0.2,
            )
            
            # Plot individual points within the hull
            ax_main.scatter(
                projected_oos_descriptors[:, 0],
                projected_oos_descriptors[:, 1],
                s=50,
                alpha=1.0,
                color=oos_colors if oos_colors is not None else "green",
                marker="x",
            )
        else:
            scatter_points = ax_main.scatter(
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
                    text = ax_main.text(
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

    # Set labels and title
    ax_main.set_title("UMAP Visualization of Sky Finder Test Set Sky Image Descriptors")
    ax_main.set_xlabel("UMAP Dimension 1")
    ax_main.set_ylabel("UMAP Dimension 2")
    ax_main.grid(True, alpha=0.3)

    # Add legend if color labels are provided
    if colors is not None and color_labels is not None:
        unique_colors = set(color_labels.keys())
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in unique_colors]
        labels = [color_labels[color] for color in unique_colors]
        plt.legend(handles, labels, title="Sky Class", loc="upper right")

    # Add interactive functionality
    if interactive:
        def on_hover(event):
            if event.inaxes == ax_main:
                # Find the closest point
                if event.xdata is not None and event.ydata is not None:
                    distances = np.sqrt((projected_descriptors[:, 0] - event.xdata)**2 + 
                                      (projected_descriptors[:, 1] - event.ydata)**2)
                    closest_idx = np.argmin(distances)
                    
                    if distances[closest_idx] < 0.1:
                        load_and_display_image(closest_idx)
        
        def load_and_display_image(idx):
            """Load and display image only when needed."""
            try:
                image_path = image_paths[idx]
                if os.path.exists(image_path):
                    # Clear previous image
                    ax_image.clear()
                    ax_image.set_title(f"Point {idx}: {os.path.basename(image_path)}")
                    ax_image.axis('off')
                    
                    # Load and display image
                    try:
                        img = mpimg.imread(image_path)
                        ax_image.imshow(img)
                        fig.canvas.draw_idle()
                    except Exception as e:
                        ax_image.text(0.5, 0.5, f"Error loading image\\n{str(e)}", ha='center', va='center', transform=ax_image.transAxes)
                        fig.canvas.draw_idle()
                else:
                    ax_image.clear()
                    ax_image.set_title("Image not found")
                    ax_image.text(0.5, 0.5, f"File not found:\\n{image_path}", ha='center', va='center', transform=ax_image.transAxes)
                    ax_image.axis('off')
                    fig.canvas.draw_idle()
            except Exception as e:
                print(f"Error displaying image {idx}: {e}")
        
        # Connect events
        fig.canvas.mpl_connect('motion_notify_event', on_hover)

    plt.tight_layout()
    plt.show()