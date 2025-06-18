import os
import sys
import torch
import argparse
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.sky_image_descriptor import (
    get_sky_finder_descriptors,
    get_fitted_umap_reducer,
    get_kmeans_groups,
    plot_sky_finder_sky_image_descriptors,
)


def load_descriptors() -> Tuple[np.ndarray, List[float], List[str], List[str]]:
    """
    Load the sky finder sky image descriptors and metadata.

    Returns:
        np.ndarray: The sky image descriptors for the Sky Finder test set.
        List[float]: The cloud coverages for the Sky Finder test set.
        List[str]: The sky classes for the Sky Finder test set.
        List[str]: The image paths for the Sky Finder test set.
    """
    try:
        sky_finder_sky_image_descriptors, cloud_coverages, sky_types, image_paths = get_sky_finder_descriptors()
        print(f"✅ Successfully loaded {len(sky_finder_sky_image_descriptors)} sky image descriptors.")
        return sky_finder_sky_image_descriptors, cloud_coverages, sky_types, image_paths
    except Exception as e:
        print(f"❌ Failed to load sky image descriptors: {e}")
        raise


def create_umap_reducer(sky_image_descriptors: np.ndarray) -> UMAP:
    """
    Create and fit a UMAP reducer for dimensionality reduction.

    Args:
        sky_image_descriptors (np.ndarray): The sky image descriptors to fit the UMAP on.

    Returns:
        UMAP: The fitted UMAP reducer.
    """
    try:
        fitted_umap_reducer = get_fitted_umap_reducer(
            sky_finder_sky_image_descriptors=sky_image_descriptors
        )
        print("✅ Successfully created and fitted UMAP reducer.")
        return fitted_umap_reducer
    except Exception as e:
        print(f"❌ Failed to create UMAP reducer: {e}")
        raise


def generate_sky_type_colors(sky_types: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate colors and labels for sky type coloring.

    Args:
        sky_types (List[str]): List of sky type labels.

    Returns:
        Tuple[List[str], Dict[str, str]]: (colors, color_labels)
    """
    try:
        colors = [
            "blue" if sky_type == "clear" else
            "red" if sky_type == "overcast" else "orange"
            for sky_type in sky_types
        ]
        color_labels = {
            "blue": "clear",
            "orange": "partial",
            "red": "overcast"
        }
        print(f"✅ Successfully generated sky type colors for {len(set(sky_types))} unique sky types.")
        return colors, color_labels
    except Exception as e:
        print(f"❌ Failed to generate sky type colors: {e}")
        raise


def generate_cluster_colors(sky_image_descriptors: np.ndarray, k: int) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate colors and labels for cluster grouping using K-means.

    Args:
        sky_image_descriptors (np.ndarray): The sky image descriptors to cluster.
        k (int): Number of clusters for K-means.

    Returns:
        Tuple[List[str], Dict[str, str]]: (colors, color_labels)
    """
    try:
        cluster_labels = get_kmeans_groups(sky_image_descriptors=sky_image_descriptors, k=k)
        colors = [
            f"C{label}" if label >= 0 else "black" for label in cluster_labels
        ]
        color_labels = {f"C{label}": f"Cluster {label}" for label in set(cluster_labels) if label >= 0}
        unique_clusters = len(set(cluster_labels))
        print(f"✅ Successfully generated cluster colors for {unique_clusters} clusters using K={k}.")
        return colors, color_labels
    except Exception as e:
        print(f"❌ Failed to generate cluster colors: {e}")
        raise


def generate_cloud_cover_colors(cloud_coverages: List[float]) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate colors and labels for cloud cover coloring.

    Args:
        cloud_coverages (List[float]): List of cloud coverages.

    Returns:
        Tuple[List[str], Dict[str, str]]: (colors, color_labels)
    """
    try:
        cmap = plt.cm.get_cmap("viridis")
        colors = [cmap(torch.sigmoid(torch.tensor(cc)).item()) for cc in cloud_coverages]
        hex_colors = [
            "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), 
                int(color[1] * 255), 
                int(color[2] * 255)
            ) 
            for color in colors
        ]

        color_labels = {}
        for cc_val in [0.0, 1.0]:
            color = cmap(cc_val)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), 
                int(color[1] * 255), 
                int(color[2] * 255)
            )
            color_labels[hex_color] = f"CC: {cc_val:.1f}"

            print(f"✅ Successfully generated cloud cover colors for {len(cloud_coverages)} values.")
        return hex_colors, color_labels
    except Exception as e:
        print(f"❌ Failed to generate cloud cover colors: {e}")
        raise


def create_visualization(
    fitted_umap: UMAP,
    sky_image_descriptors: np.ndarray,
    colors: List[str],
    color_labels: Dict[str, str],
    image_paths: List[str],
    interactive: bool,
) -> None:
    """
    Create and display the sky image descriptor visualization.

    Args:
        fitted_umap (UMAP): The fitted UMAP reducer.
        sky_image_descriptors (np.ndarray): The sky image descriptors.
        colors (List[str]): Colors for each data point.
        color_labels (Dict[str, str]): Mapping of colors to labels.
        image_paths (List[str]): Paths to the images.
        interactive (bool): Whether to run in interactive mode.
    """
    try:
        print("▶️  Creating sky image descriptor visualization...")
        plot_sky_finder_sky_image_descriptors(
            fitted_umap=fitted_umap,
            sky_finder_sky_image_descriptors=sky_image_descriptors,
            colors=colors,
            color_labels=color_labels,
            image_paths=image_paths if interactive else None,
        )
        print("✅ Successfully created sky image descriptor visualization.")
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot sky image descriptor space for sky finder dataset using UMAP dimensionality reduction."
    )
    
    parser.add_argument(
        "-c",
        "--color-by",
        type=str,
        choices=["sky_type", "cluster", "cloud_cover"],
        default="sky_type",
        help="Coloring method for visualization: 'sky_type' for semantic labels, 'cluster' for K-means clustering or 'cloud_cover' for cloud cover percentage (default: sky_type).",
    )
    
    parser.add_argument(
        "-k",
        "--k-clusters",
        type=int,
        default=3,
        help="Number of clusters for K-means when using cluster grouping (default: 3)",
    )
    
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enable interactive mode with image tooltips on hover",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to plot sky image descriptor space.
    """
    args = parse_args()

    print("▶️  Starting sky image descriptor space visualization...")
    print(f"📋 Configuration:")
    print(f"   • Coloring method: {args.color_by}")
    print(f"   • K-means clusters: {args.k_clusters}")
    print(f"   • Interactive mode: {args.interactive}")

    try:
        # Load sky image descriptors
        print("▶️  Loading sky image descriptors...")
        sky_image_descriptors, cloud_coverages, sky_types, image_paths = load_descriptors()

        # Create UMAP reducer
        print("▶️  Creating UMAP dimensionality reduction...")
        fitted_umap_reducer = create_umap_reducer(sky_image_descriptors)

        # Generate colors based on coloring method
        if args.color_by == "sky_type":
            print("▶️  Generating colors for sky type coloring...")
            colors, color_labels = generate_sky_type_colors(sky_types)
        elif args.color_by == "cluster":
            print(f"▶️  Generating colors for cluster coloring (K={args.k_clusters})...")
            colors, color_labels = generate_cluster_colors(sky_image_descriptors, args.k_clusters)
        elif args.color_by == "cloud_cover":
            print("▶️  Generating colors for cloud cover coloring...")
            colors, color_labels = generate_cloud_cover_colors(cloud_coverages)

        # Create visualization
        create_visualization(
            fitted_umap=fitted_umap_reducer,
            sky_image_descriptors=sky_image_descriptors,
            colors=colors,
            color_labels=color_labels,
            image_paths=image_paths,
            interactive=args.interactive,
        )

        print("🎉 Sky image descriptor visualization completed successfully!")

    except Exception as e:
        print(f"💥 Fatal error during visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()