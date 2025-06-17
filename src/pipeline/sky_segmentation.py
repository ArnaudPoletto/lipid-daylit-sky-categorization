import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gsam2")))

from src.gsam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.gsam2.sam2.build_sam import build_sam2
from src.config import DEVICE


def get_models(
    sam2_type: str = "large",
    gdino_type: str = "tiny",
) -> Tuple[SAM2ImagePredictor, AutoProcessor, AutoModelForZeroShotObjectDetection]:
    """
    Load the pretrained SAM2 model, Grounding DINO processor, and Grounding DINO model.
    
    This function initializes the segmentation pipeline by loading both SAM2 (Segment Anything Model 2)
    and Grounding DINO models. SAM2 is used for generating high-quality segmentation masks, while
    Grounding DINO provides zero-shot object detection capabilities to identify sky regions.

    Args:
        sam2_type (str): Type of SAM2 model to use. Options are 'tiny', 'small', 'base', or 'large'.
            Larger models provide better accuracy but require more computational resources.
        gdino_type (str): Type of Grounding DINO model to use. Options are 'tiny' or 'base'.
            The base model offers better detection performance at the cost of speed.

    Returns:
        Tuple containing:
            - SAM2ImagePredictor: Initialized SAM2 image predictor for mask generation.
            - AutoProcessor: Initialized Grounding DINO processor for input preprocessing.
            - AutoModelForZeroShotObjectDetection: Initialized Grounding DINO model for sky detection.

    Raises:
        ValueError: If an invalid SAM2 type is specified.
        ValueError: If an invalid Grounding DINO type is specified.
    """
    # Validate model type arguments
    if sam2_type not in ["tiny", "small", "base", "large"]:
        raise ValueError(
            f"❌ Invalid SAM2 type: {sam2_type}. Choose from 'tiny', 'small', 'base', or 'large'."
        )
    if gdino_type not in ["tiny", "base"]:
        raise ValueError(
            f"❌ Invalid GDINO type: {gdino_type}. Choose from 'tiny' or 'base'."
        )

    # Configure SAM2 model paths based on selected type
    sam2_configs = {
        "tiny": {
            "checkpoint": "./../../src/gsam2/checkpoints/sam2.1_hiera_tiny.pt",
            "config": "./configs/sam2.1/sam2.1_hiera_t.yaml"
        },
        "small": {
            "checkpoint": "./../../src/gsam2/checkpoints/sam2.1_hiera_small.pt",
            "config": "./configs/sam2.1/sam2.1_hiera_s.yaml"
        },
        "base": {
            "checkpoint": "./../../src/gsam2/checkpoints/sam2.1_hiera_base_plus.pt",
            "config": "./configs/sam2.1/sam2.1_hiera_b+.yaml"
        },
        "large": {
            "checkpoint": "./../../src/gsam2/checkpoints/sam2.1_hiera_large.pt",
            "config": "./configs/sam2.1/sam2.1_hiera_l.yaml"
        }
    }

    # Initialize SAM2 model
    sam2_checkpoint = sam2_configs[sam2_type]["checkpoint"]
    model_cfg = sam2_configs[sam2_type]["config"]
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    print(f"✅ Initialized SAM2 {sam2_type} model.")

    # Initialize Grounding DINO processor and model
    model_id = f"IDEA-Research/grounding-dino-{gdino_type}"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
    print(f"✅ Initialized Grounding DINO {gdino_type} model.")

    return image_predictor, grounding_processor, grounding_model


def get_sky_mask(
    frame: np.ndarray,
    image_predictor: SAM2ImagePredictor,
    grounding_processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Get the sky mask from the given frame using SAM2 and Grounding DINO.
    
    This function performs sky segmentation in two stages:
    1. Uses Grounding DINO to detect sky regions in the image through zero-shot object detection
    2. Uses SAM2 to generate precise segmentation masks for the detected sky regions
    
    The resulting mask is post-processed using morphological operations to clean up the boundaries
    and remove small artifacts.

    Args:
        frame (np.ndarray): The input frame as a NumPy array in BGR format.
        image_predictor (SAM2ImagePredictor): SAM2 image predictor model for mask generation.
        grounding_processor (AutoProcessor): Grounding DINO processor for input preprocessing.
        grounding_model (AutoModelForZeroShotObjectDetection): Grounding DINO model for sky detection.
        box_threshold (float): Threshold for box detection confidence (0-1). 
            Lower values detect more regions but may include false positives.
        text_threshold (float): Threshold for text-to-region matching confidence (0-1).
            Lower values allow weaker text-region associations.

    Returns:
        Tuple containing:
            - np.ndarray: Binary sky mask where True/1 indicates sky pixels.
            - Tuple[int, int, int, int]: Bounding box of the sky region (x_min, y_min, x_max, y_max).

    Raises:
        ValueError: If no sky region is detected in the image.
    """
    # Convert frame to PIL Image for Grounding DINO
    text = "sky."
    image = Image.fromarray(frame)

    # Run Grounding DINO to detect sky regions
    inputs = grounding_processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Post-process detection results
    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    # Check if any sky regions were detected
    if results[0]["scores"].numel() == 0:
        raise ValueError(
            f"❌ No sky detected. Make sure the image contains a visible sky and try again with "
            f"different box and text thresholds. Current thresholds are box: {box_threshold}, text: {text_threshold}."
        )

    # Use SAM2 to generate segmentation masks for detected boxes
    image_predictor.set_image(np.array(image.convert("RGB")))
    input_boxes = results[0]["boxes"].cpu().numpy()
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Flatten scores and squeeze masks if needed
    scores = scores.flatten()
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Merge all detected sky masks into a single mask
    sky_mask = np.zeros_like(masks[0, :, :], dtype=bool)
    n_objects = scores.shape[0]
    for i in range(n_objects):
        mask = masks[i, :, :].astype(bool)
        sky_mask |= mask

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.erode(sky_mask.astype(np.uint8), kernel, iterations=1)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

    # Calculate bounding box for the sky region
    y_indices, x_indices = np.where(sky_mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Return full frame bounds if no valid mask pixels found
        return sky_mask, (0, 0, sky_mask.shape[1], sky_mask.shape[0])

    # Get tight bounding box coordinates
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    sky_bounding_box = (x_min, y_min, x_max + 1, y_max + 1)
    
    # Crop mask to bounding box
    sky_mask = sky_mask[y_min : y_max + 1, x_min : x_max + 1]

    return sky_mask, sky_bounding_box