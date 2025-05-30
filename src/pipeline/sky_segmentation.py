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
    Initialize and return the SAM2 model, the Grounding DINO processor, and the Grounding DINO model.

    Args:
        sam2_type (str): Type of SAM2 model to use. Options are 'tiny', 'small', 'base', or 'large'.
        gdino_type (str): Type of Grounding DINO model to use. Options are 'tiny' or 'base'.

    Raises:
        ValueError: If an invalid SAM2 type is specified.
        ValueError: If an invalid Grounding DINO type is specified.

    Returns:
        SAM2ImagePredictor: Initialized SAM2 image predictor.
        AutoProcessor: Initialized Grounding DINO processor.
        AutoModelForZeroShotObjectDetection: Initialized Grounding DINO model.
    """
    if sam2_type not in ["tiny", "small", "base", "large"]:
        raise ValueError(
            f"❌ Invalid SAM2 type: {sam2_type}. Choose from 'tiny', 'small', 'base', or 'large'."
        )
    if gdino_type not in ["tiny", "base"]:
        raise ValueError(
            f"❌ Invalid GDINO type: {gdino_type}. Choose from 'tiny' or 'base'."
        )

    # Initialize SAM model
    if sam2_type == "tiny":
        sam2_checkpoint = "./../../src/gsam2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"
    elif sam2_type == "small":
        sam2_checkpoint = "./../../src/gsam2/checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"
    elif sam2_type == "base":
        sam2_checkpoint = "./../../src/gsam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif sam2_type == "large":
        sam2_checkpoint = "./../../src/gsam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    print(f"✅ Initialized SAM2 {sam2_type} model.")

    # Initialize GDINO processor and model
    model_id = f"IDEA-Research/grounding-dino-{gdino_type}"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        DEVICE
    )
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

    Args:
        frame (np.ndarray): The input frame as a NumPy array.
        image_predictor (SAM2ImagePredictor): SAM2 image predictor model.
        grounding_processor (AutoProcessor): Grounding DINO processor.
        grounding_model (AutoModelForZeroShotObjectDetection): Grounding DINO model.
        box_threshold (float): Threshold for box detection.
        text_threshold (float): Threshold for text detection.

    Returns:
        np.ndarray: The sky mask as a NumPy array.
        Tuple[int, int, int, int]: The bounding box of the sky region in the format (x_min, y_min, x_max, y_max).
    """
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    text = "sky."
    image = Image.fromarray(frame)

    # Run Grounding DINO on the frame
    inputs = grounding_processor(images=image, text=text, return_tensors="pt").to(
        DEVICE
    )
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    # Check if no objects were detected and exit early
    if results[0]["scores"].numel() == 0:
        raise ValueError(
            f"❌ No sky detected. Make sure the image contains a visible sky and try again with different box and text thresholds. Current thresholds are box: {box_threshold}, text: {text_threshold}."
        )

    # Prompt SAM 2 image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))
    input_boxes = results[0]["boxes"].cpu().numpy()
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    scores = scores.flatten()
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Merge all masks found together
    sky_mask = np.zeros_like(masks[0, :, :], dtype=bool)
    n_objects = scores.shape[0]
    for i in range(n_objects):
        mask = masks[i, :, :].astype(bool)
        sky_mask |= mask

    # Process the mask by dilating the sky region
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.erode(sky_mask.astype(np.uint8), kernel, iterations=1)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

    # Get bounding box and crop the sky mask
    y_indices, x_indices = np.where(sky_mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, sky_mask.shape[0], sky_mask.shape[1], 0]

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    sky_bounding_box = [x_min, y_min, x_max + 1, y_max + 1]
    sky_mask = sky_mask[y_min : y_max + 1, x_min : x_max + 1]

    return sky_mask, sky_bounding_box
