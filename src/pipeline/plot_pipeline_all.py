import os
import re
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path
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


def get_processed_video_files() -> List[str]:
    """
    Get all processed video JSON files from the specified directory.

    Returns:
        List[str]: List of JSON file paths.
    """
    if not os.path.exists(GENERATED_PIPELINE_PATH):
        raise FileNotFoundError(
            f"❌ Generated pipeline directory not found at {os.path.abspath(GENERATED_PIPELINE_PATH)}"
        )

    # Look for JSON files with the pattern *_processed.json
    json_pattern = os.path.join(GENERATED_PIPELINE_PATH, "*_processed.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        raise ValueError(
            f"❌ No processed video JSON files found in {os.path.abspath(GENERATED_PIPELINE_PATH)}"
        )

    json_files.sort()  # Sort for consistent ordering
    return json_files


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

        return video_data
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format in {json_file_path}: {e}")
    except Exception as e:
        raise IOError(f"❌ Could not read {json_file_path}: {e}")


def collect_data(json_files: List[str]) -> tuple:
    """
    Collect data from all processed video JSON files.

    Args:
        json_files (List[str]): List of JSON file paths.

    Returns:
        tuple: (texture_descriptors_array, video_names_list)
    """
    all_texture_descriptors = []
    sky_classes = []
    video_names = []

    print(f"📖 Loading texture descriptors from {len(json_files)} video(s)...")

    for json_file in json_files:
        try:
            video_data = load_video_data(json_file)
            mean_texture_descriptor = np.array(video_data["mean_texture_descriptor"])
            sky_class = video_data.get("sky_class")
            all_texture_descriptors.append(mean_texture_descriptor)
            sky_classes.append(sky_class)

            # Extract video name from filename (remove _processed.json)
            video_name = (
                os.path.basename(json_file)
                .replace("_processed.json", "")
                .replace("_", ".")
            )
            video_names.append(video_name)

        except Exception as e:
            print(f"⚠️  Skipping {os.path.basename(json_file)}: {e}")
            continue

    return np.array(all_texture_descriptors), np.array(sky_classes), video_names


def plot_all_videos(
    all_texture_descriptors: np.ndarray,
    all_sky_classes: np.ndarray,
    video_names: List[str],
) -> None:
    """
    Plot all videos' texture descriptors in the UMAP space.

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

    # Get Out-of-Sample labels
    oos_labels = []
    for video_name in video_names:
        # Remove file extensions
        video_name = re.sub(r'\.(mp4|avi|mov|mkv)$', '', video_name, flags=re.IGNORECASE)
        
        # Check patterns and convert
        if re.match(r'P\d+Scene\d+', video_name):
            # PXSceneYY -> X|YY
            match = re.match(r'P(\d+)Scene(\d+)', video_name)
            oos_label = f"{match.group(1)}|{match.group(2)}"
        elif re.match(r'P\d+Clear\d+', video_name):
            # PXClearYY -> Xc|YY
            match = re.match(r'P(\d+)Clear(\d+)', video_name)
            oos_label = f"{match.group(1)}c|{match.group(2)}"
        elif re.match(r'P\d+Overcast\d+', video_name):
            # PXOvercastYY -> Xo|YY
            match = re.match(r'P(\d+)Overcast(\d+)', video_name)
            oos_label = f"{match.group(1)}o|{match.group(2)}"
        else:
            oos_label = video_name
        oos_labels.append(oos_label)

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
        oos_labels=oos_labels,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot texture descriptors for all processed videos in UMAP space."
    )

    parser.add_argument(
        "--list-videos",
        "-lv",
        action="store_true",
        help="List all available processed videos and exit.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to plot all processed videos' texture descriptors.
    """
    args = parse_args()

    # Get all processed video JSON files
    json_files = get_processed_video_files()
    print(f"➡️  Found {len(json_files)} processed video file(s)")

    # If user just wants to list videos, do that and exit
    if args.list_videos:
        print("\n📋 Available processed videos:")
        for i, json_file in enumerate(json_files, 1):
            video_name = (
                os.path.basename(json_file)
                .replace("_processed.json", "")
                .replace("_", ".")
            )
            print(f"  {i:2d}. {video_name}")
        return

    # Collect all texture descriptors
    all_texture_descriptors, all_sky_classes, video_names = collect_data(json_files)

    # Plot all videos
    plot_all_videos(
        all_texture_descriptors=all_texture_descriptors,
        all_sky_classes=all_sky_classes,
        video_names=video_names,
    )


if __name__ == "__main__":
    main()
