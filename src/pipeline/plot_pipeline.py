import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.texture_descriptor import (
    get_sky_finder_texture_descriptors,
    get_fitted_umap_reducer,
    plot_sky_finder_texture_descriptors,
)
from src.config import (
    GENERATED_PIPELINE_PATH,
)

def get_processed_video_file_path(
    video_path: str,
) -> str:
    """
    Get processed video JSON file path based on the video file path.

    Returns:
        str: Path to the processed video JSON file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"❌ Video file not found at {os.path.abspath(video_path)}."
        )
    json_file_path = os.path.join(
        GENERATED_PIPELINE_PATH,
        f"{os.path.basename(video_path).replace('.', '_')}_processed.json"
    )
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(
            f"❌ Processed video JSON file not found at {os.path.abspath(json_file_path)}."
        )
    
    return json_file_path


def load_video_data(json_file_path: str) -> Dict[str, Any]:
    """
    Load processed video data from JSON file.

    Args:
        json_file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Loaded video data.
    """
    try:
        with open(json_file_path, "r") as f:
            video_data = json.load(f)

        if "mean_texture_descriptor" not in video_data:
            raise ValueError(
                f"❌ Mean texture descriptor not found in {json_file_path}"
            )
        if "majority_sky_class" not in video_data:
            raise ValueError(
                f"❌ Majority sky class not found in {json_file_path}"
            )

        return video_data
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format in {json_file_path}: {e}")
    except Exception as e:
        raise IOError(f"❌ Could not read {json_file_path}: {e}")
    

def collect_data(
        json_file_path: str,
        plot_time: bool,
    ) -> tuple:
    """
    Collect data from processed video JSON file path.

    Args:
        json_file_paths (List[str]): List of JSON file paths.
        plot_time (bool): Whether to plot texture descriptors over time.

    Returns:
        tuple: (texture_descriptors_array, sky_classes)
    """
    video_data = load_video_data(json_file_path)
    if plot_time:
        texture_descriptors = []
        sky_classes = []
        for key, descriptors in video_data.items():
            if key in ["mean_texture_descriptor", "majority_sky_class"]:
                continue

            texture_descriptor = np.array(descriptors["texture_descriptor"])
            sky_class = descriptors["sky_class"]
            texture_descriptors.append(texture_descriptor)
            sky_classes.append(sky_class)
        texture_descriptors = np.array(texture_descriptors)
        sky_classes = np.array(sky_classes)
    else:
        texture_descriptors = np.array([video_data["mean_texture_descriptor"]])
        sky_classes = np.array([video_data.get("majority_sky_class")])

    return texture_descriptors, sky_classes


def plot_video(
    all_texture_descriptors: np.ndarray,
    all_sky_classes: np.ndarray,
    labels: List[str],
) -> None:
    """
    Plot video's texture descriptors in the UMAP space.

    Args:
        all_texture_descriptors (np.ndarray): Array of all texture descriptors.
        all_sky_classes (np.ndarray): Array of sky classes corresponding to descriptors.
        video_names (List[str]): List of video names corresponding to descriptors.
    """
    print("➡️  Generating UMAP visualization...")

    # Get sky finder texture descriptors and fitted UMAP reducer
    sky_finder_texture_descriptors, sky_types = get_sky_finder_texture_descriptors()
    colors = [
        "blue" if sky_type == "clear" else
        "red" if sky_type == "overcast" else "orange"
        for sky_type in sky_types
    ]
    fitted_umap_reducer = get_fitted_umap_reducer(
        sky_finder_texture_descriptors=sky_finder_texture_descriptors
    )

    # Get Out-of-Sample colors
    oos_colors = []
    for sky_class in all_sky_classes:
        oos_color = ["blue", "orange", "red"][sky_class]
        oos_colors.append(oos_color)

    # Plot all videos in the UMAP space
    plot_sky_finder_texture_descriptors(
        fitted_umap=fitted_umap_reducer,
        sky_finder_texture_descriptors=sky_finder_texture_descriptors,
        colors=colors,
        oos_texture_descriptors=all_texture_descriptors,
        oos_colors=oos_colors,
        oos_labels=labels,
        oos_as_convex_hull=True,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot texture descriptors for all processed videos in UMAP space."
    )

    parser.add_argument(
        "--video-path",
        "-vp",
        type=str,
        required=True,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--plot-time",
        "-pt",
        action="store_true",
        help="Plot the each video's texture descriptor over time.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to plot a video's texture descriptors.
    """
    args = parse_args()
    video_path = args.video_path
    plot_time = args.plot_time

    json_file_path = get_processed_video_file_path(video_path)
    texture_descriptors, sky_classes = collect_data(json_file_path, plot_time)
    video_name = os.path.basename(json_file_path).replace("_processed.json", "").replace("_", ".")

    plot_video(
        all_texture_descriptors=texture_descriptors,
        all_sky_classes=sky_classes,
        labels=[video_name],
    )

if __name__ == "__main__":
    main()