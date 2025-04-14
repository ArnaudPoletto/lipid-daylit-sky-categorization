import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.file import get_paths_recursive
from src.config import (
    SKY_FINDER_MASKS_PATH,
    SKY_FINDER_IMAGES_PATH,
    SKY_FINDER_SKY_CLASSES,
)


def get_sky_finder_paths_dict() -> Dict[str, Dict[str, List[str]]]:
    """
    Initialize the paths dictionary for the dataset.
    The dictionary contains a 3-level structure:
    - Level 1: Sky class, either clear, partial or overcast
    - Level 2: Folder name, corresponding to the sky finder dataset camera identifier
    - Level 3: List of image paths

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary with sky classes as keys and dictionaries of folder names and image paths as values.
    """
    paths_dict = {}
    for sky_class in SKY_FINDER_SKY_CLASSES:
        paths_dict[sky_class] = {}
        sky_finder_class_images_path = f"{SKY_FINDER_IMAGES_PATH}/{sky_class}"
        folder_names = os.listdir(sky_finder_class_images_path)
        for folder_name in folder_names:
            folder_path = os.path.join(sky_finder_class_images_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            image_paths = get_paths_recursive(
                folder_path=folder_path,
                match_pattern="*.jpg",
                path_type="f",
                recursive=True,
            )
            paths_dict[sky_class][folder_name] = image_paths

    return paths_dict


def get_sky_finder_camera_ids(paths_dict: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """
    Get the camera IDs from the dataset.

    Args:
        paths_dict (Dict[str, Dict[str, List[str]]]): A dictionary with sky classes as keys and camera IDs as values.

    Returns:
        List[str]: A list of camera IDs.
    """
    camera_ids = []
    for sky_class in paths_dict.keys():
        camera_ids.extend(paths_dict[sky_class].keys())
    unique_camera_ids = list(set(camera_ids))

    return unique_camera_ids


def get_sky_finder_masks(
    paths_dict: Dict[str, Dict[str, List[str]]],
) -> Dict[str, np.ndarray]:
    """
    Get the camera masks from the dataset, segmenting the sky from the ground.

    Args:
        paths_dict (Dict[str, Dict[str, List[str]]]): A dictionary with sky classes as keys and camera IDs as values.

    Returns:
            Dict[str, np.ndarray]: A dictionary with camera IDs as keys and masks as values.
    """
    camera_ids = get_sky_finder_camera_ids(paths_dict)
    masks = {}
    for camera_id in camera_ids:
        mask_path = f"{SKY_FINDER_MASKS_PATH}/{camera_id}.png"
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask > 128
            masks[camera_id] = mask
        else:
            print(f"❌ Mask not found for camera ID {camera_id}.")

    print(f"✅ Found {len(masks)} masks for {len(camera_ids)} camera IDs.")

    return masks


def get_sky_finder_bounding_boxes(
    paths_dict: Dict[str, Dict[str, List[str]]],
    ground_percentage_threshold: float = 0.1,
    min_height: int = 50,
    crop_step: int = 1,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Get the bounding boxes for the cameras based on the masks.

    Args:
        paths_dict (Dict[str, Dict[str, List[str]]]): A dictionary with sky classes as keys and camera IDs as values.
        ground_percentage_threshold (float): The threshold for the ground percentage.
        min_height (int): The minimum height for the bounding box.
        crop_step (int): The step size for the binary search.

    Returns:
        Dict[str, Tuple[int, int, int, int]]: A dictionary with camera IDs as keys and bounding boxes as values.
    """
    masks = get_sky_finder_masks(paths_dict)
    bounding_boxes = {}
    for camera_id, mask in masks.items():
        height, width = mask.shape
        y_min, y_max = 0, height

        # Start with full image
        ground_pixels = np.sum(~mask)
        total_pixels = width * height
        ground_percentage = ground_pixels / total_pixels
        while True:
            # Keep cropping until we're below threshold or can't crop anymore
            if ground_percentage <= ground_percentage_threshold:
                break
            if (y_max - y_min) <= min_height:
                break
            if y_min + crop_step >= y_max - crop_step:
                break

            # Get top part
            top_rows = mask[y_min : y_min + crop_step, :]
            top_ground_pixels = np.sum(~top_rows)
            top_density = top_ground_pixels / (width * crop_step)
            # Get bottom part
            bottom_rows = mask[y_max - crop_step : y_max, :]
            bottom_ground_pixels = np.sum(~bottom_rows)
            bottom_density = bottom_ground_pixels / (width * crop_step)

            # Select the part with the most sky and update the bounding box
            total_pixels -= width * crop_step
            if top_density > bottom_density:
                ground_pixels -= top_ground_pixels
                y_min += crop_step
            else:
                ground_pixels -= bottom_ground_pixels
                y_max -= crop_step

            ground_percentage = ground_pixels / total_pixels

        # Set the bounding box
        bounding_boxes[camera_id] = (0, y_min, width, y_max)

    print(f"✅ Found bounding boxes for {len(bounding_boxes)} camera IDs.")

    return bounding_boxes
