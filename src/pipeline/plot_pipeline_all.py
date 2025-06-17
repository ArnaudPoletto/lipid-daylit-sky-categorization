import os
import re
import sys
import json
import glob
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.sky_image_descriptor import (
    get_sky_finder_descriptors,
    get_fitted_umap_reducer,
    plot_sky_finder_sky_image_descriptors,
)
from src.config import (
    GENERATED_PIPELINE_PATH,
)


def get_processed_video_file_paths() -> List[str]:
    """
    Get all processed video JSON file paths from the generated pipeline directory.
    
    This function searches for JSON files that contain processed video analysis results,
    specifically those with the pattern '*_processed.json'.

    Returns:
        List[str]: Sorted list of paths to processed video JSON files.
        
    Raises:
        FileNotFoundError: If the generated pipeline directory does not exist.
        ValueError: If no processed video files are found in the directory.
    """
    if not os.path.exists(GENERATED_PIPELINE_PATH):
        raise FileNotFoundError(
            f"❌ Generated pipeline directory not found at {os.path.abspath(GENERATED_PIPELINE_PATH)}"
        )

    # Search for processed video JSON files
    json_pattern = os.path.join(GENERATED_PIPELINE_PATH, "*_processed.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        raise ValueError(
            f"❌ No processed video JSON files found in {os.path.abspath(GENERATED_PIPELINE_PATH)}"
        )

    # Sort for consistent ordering across runs
    json_files.sort()
    
    return json_files


def load_video_data(json_file_path: str) -> Dict[str, Any]:
    """
    Load processed video data from a JSON file.
    
    This function reads and validates the processed video analysis results,
    ensuring that required fields are present in the data.

    Args:
        json_file_path (str): Path to the JSON file containing video analysis results.

    Returns:
        Dict[str, Any]: Dictionary containing the loaded video analysis data.
        
    Raises:
        ValueError: If required fields are missing or JSON format is invalid.
        IOError: If the file cannot be read.
    """
    try:
        with open(json_file_path, "r") as f:
            video_data = json.load(f)

        # Validate required fields
        if "mean_sky_image_descriptor" not in video_data:
            raise ValueError(
                f"❌ Mean sky image descriptor not found in {json_file_path}"
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


def collect_video_data(json_file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect sky image descriptors and metadata from all processed video files.
    
    This function aggregates data from multiple processed video files, extracting
    the mean sky image descriptors, sky classifications, and video identifiers.

    Args:
        json_file_paths (List[str]): List of paths to processed video JSON files.

    Returns:
        Tuple containing:
            - np.ndarray: Array of mean sky image descriptors with shape (n_videos, descriptor_dim).
            - np.ndarray: Array of majority sky classes with shape (n_videos,).
            - List[str]: List of video names extracted from filenames.
    """
    all_sky_image_descriptors = []
    all_sky_classes = []
    video_names = []

    print(f"📖 Loading sky image descriptors from {len(json_file_paths)} video(s)...")

    for json_file_path in json_file_paths:
        # Load video analysis data
        video_data = load_video_data(json_file_path)
        
        # Extract sky image descriptor and class
        mean_sky_image_descriptor = np.array(video_data["mean_sky_image_descriptor"])
        majority_sky_class = video_data["majority_sky_class"]
        
        all_sky_image_descriptors.append(mean_sky_image_descriptor)
        all_sky_classes.append(majority_sky_class)

        # Extract video name from filename
        video_name = os.path.basename(json_file_path)
        video_name = video_name.replace("_processed.json", "").replace("_", ".")
        video_names.append(video_name)

    return (
        np.array(all_sky_image_descriptors), 
        np.array(all_sky_classes), 
        video_names
    )


def format_video_labels(video_names: List[str]) -> List[str]:
    """
    Format video names into concise labels for visualization.
    
    This function converts video filenames into shorter, more readable labels
    by applying pattern-based transformations.

    Args:
        video_names (List[str]): List of video filenames to format.

    Returns:
        List[str]: List of formatted labels for visualization.
    """
    formatted_labels = []
    
    for video_name in video_names:
        # Remove common file extensions
        clean_name = re.sub(r'\.(mp4|avi|mov|mkv)$', '', video_name, flags=re.IGNORECASE)
        
        # Apply pattern-based formatting
        if re.match(r'P\d+Scene\d+', clean_name):
            # PXSceneYY -> X|YY
            match = re.match(r'P(\d+)Scene(\d+)', clean_name)
            label = f"{match.group(1)}|{match.group(2)}"
        elif re.match(r'P\d+Clear\d+', clean_name):
            # PXClearYY -> Xc|YY
            match = re.match(r'P(\d+)Clear(\d+)', clean_name)
            label = f"{match.group(1)}c|{match.group(2)}"
        elif re.match(r'P\d+Overcast\d+', clean_name):
            # PXOvercastYY -> Xo|YY
            match = re.match(r'P(\d+)Overcast(\d+)', clean_name)
            label = f"{match.group(1)}o|{match.group(2)}"
        else:
            # Use original name if no pattern matches
            label = clean_name
            
        formatted_labels.append(label)
    
    return formatted_labels


def get_sky_class_colors(sky_classes: np.ndarray) -> List[str]:
    """
    Map sky class indices to corresponding colors for visualization.
    
    Args:
        sky_classes (np.ndarray): Array of sky class indices (0, 1, or 2).

    Returns:
        List[str]: List of color names corresponding to each sky class.
    """
    color_map = {
        0: "blue",    # Clear sky
        1: "orange",  # Partially cloudy
        2: "red"      # Overcast
    }
    
    return [color_map[sky_class] for sky_class in sky_classes]


def plot_all_videos(
    video_sky_image_descriptors: np.ndarray,
    video_sky_classes: np.ndarray,
    video_names: List[str],
) -> None:
    """
    Plot all videos' sky image descriptors in the UMAP space.
    
    This function visualizes video sky image descriptors alongside the Sky Finder
    dataset descriptors in a 2D UMAP projection, allowing for visual comparison
    of video characteristics against the reference dataset.

    Args:
        video_sky_image_descriptors (np.ndarray): Array of video sky image descriptors.
        video_sky_classes (np.ndarray): Array of sky classes for each video.
        video_names (List[str]: List of video names for labeling.
    """
    print("▶️  Generating UMAP visualization...")

    # Load Sky Finder reference data
    sky_finder_descriptors, _, sky_classes, _ = get_sky_finder_descriptors()
    
    # Create color mapping for Sky Finder data
    sky_finder_colors = get_sky_class_colors([
        0 if sky_class == "clear" else 2 if sky_class == "overcast" else 1
        for sky_class in sky_classes
    ])
    
    # Fit UMAP on Sky Finder descriptors
    fitted_umap_reducer = get_fitted_umap_reducer(
        sky_finder_sky_image_descriptors=sky_finder_descriptors
    )

    # Format video labels and colors
    video_labels = format_video_labels(video_names)
    video_colors = get_sky_class_colors(video_sky_classes)

    # Create visualization
    plot_sky_finder_sky_image_descriptors(
        fitted_umap=fitted_umap_reducer,
        sky_finder_sky_image_descriptors=sky_finder_descriptors,
        colors=sky_finder_colors,
        oos_sky_image_descriptors=video_sky_image_descriptors,
        oos_colors=video_colors,
        oos_labels=video_labels,
    )
    
    print("✅ Visualization complete.")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot sky image descriptors from all processed videos in UMAP space."
    )
    
    parser.add_argument(
        "--pipeline-path",
        "-p",
        type=str,
        default=None,
        help="Override path to the generated pipeline directory (default: use config value).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to plot all processed videos' sky image descriptors.
    """
    args = parse_args()
    
    print("▶️  Starting video sky image descriptor visualization...")
    
    # Override pipeline path if provided
    if args.pipeline_path:
        global GENERATED_PIPELINE_PATH
        GENERATED_PIPELINE_PATH = args.pipeline_path
        print(f"📋 Using custom pipeline path: {os.path.abspath(GENERATED_PIPELINE_PATH)}")

    try:
        # Get all processed video files
        json_file_paths = get_processed_video_file_paths()
        print(f"✅ Found {len(json_file_paths)} processed video file(s).")

        # Collect sky image descriptors and metadata
        video_descriptors, video_classes, video_names = collect_video_data(json_file_paths)
        print(f"✅ Loaded sky image descriptors from all videos.")

        # Generate visualization
        plot_all_videos(
            video_sky_image_descriptors=video_descriptors,
            video_sky_classes=video_classes,
            video_names=video_names,
        )

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()