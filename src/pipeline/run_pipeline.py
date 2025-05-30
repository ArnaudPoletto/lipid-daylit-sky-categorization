import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.sky_segmentation import get_sky_mask
from src.pipeline.texture_descriptor import get_texture_descriptor
from src.pipeline.texture_descriptor import get_model as get_texture_model
from src.pipeline.sky_segmentation import get_models as get_sky_segmentation_models

def get_video(video_path: str) -> cv2.VideoCapture:
    """
    Get a video capture object for the specified video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"❌ Video file not found at {os.path.abspath(video_path)}."
        )
    if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise ValueError(
            "❌ Unsupported video file format. Please provide a valid video file."
        )

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"❌ Could not open video file at {os.path.abspath(video_path)}.")

    video_frame_rate = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return video, video_frame_rate, video_frame_count


def process_frame(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    bounds: Optional[Tuple[int, int, int, int]],
    sky_mask: Optional[np.ndarray],
    sky_bounding_box: Optional[Tuple[int, int, int, int]],
    image_predictor: Optional[object],
    grounding_processor: Optional[object],
    grounding_model: Optional[object],
    box_threshold: float,
    text_threshold: float,
    texture_model: object,
) -> Dict[str, Any]:
    """
    Process a single video frame, applying a mask if provided and displaying it.

    Args:
        frame (np.ndarray): The video frame to process.
        mask (Optional[np.ndarray]): Binary mask to apply to the frame.
        bounds (Optional[Tuple[int, int, int, int]]): Bounding box coordinates to crop the frame.
        texture_model (Optional[object]): Texture descriptor model.
        sky_mask (Optional[np.ndarray]): Precomputed sky mask for the frame.
        sky_bounding_box (Optional[Tuple[int, int, int, int]]): Bounding box for the sky mask.
        image_predictor (Optional[object]): SAM2 image predictor model.
        grounding_processor (Optional[object]): Grounding DINO processor.
        box_threshold (float): Threshold for sky box detection.
        text_threshold (float): Threshold for sky text detection.
        grounding_model (object): Grounding DINO model.

    Returns:
        Dict[str, Any]: A dictionary containing the sky mask, bounding box, and descriptors.
    """
    # Apply the first mask if provided
    if mask is not None:
        frame *= mask[:, :, np.newaxis]
        frame = frame[bounds[1] : bounds[3], bounds[0] : bounds[2]]

    # Get sky mask if models are provided
    compute_sky_mask = (
        sky_mask is None
        and sky_bounding_box is None
        and image_predictor is not None
        and grounding_processor is not None
        and grounding_model is not None
    )
    if compute_sky_mask:
        sky_mask, sky_bounding_box = get_sky_mask(
            frame=frame,
            image_predictor=image_predictor,
            grounding_processor=grounding_processor,
            grounding_model=grounding_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    # Apply the sky mask
    if sky_bounding_box is not None:
        x_min, y_min, x_max, y_max = sky_bounding_box
        frame = frame[y_min:y_max, x_min:x_max]
    if sky_mask is not None:
        frame *= sky_mask[:, :, np.newaxis]

    # Apply inpainting to fill the masked area
    if sky_mask is not None:
        inpaint_mask = (~(sky_mask.astype(np.bool))).astype(np.uint8) * 255
        frame = cv2.inpaint(frame, inpaint_mask, 3, cv2.INPAINT_TELEA)

    # Get texture descriptor
    texture_descriptor = get_texture_descriptor(
        frame=frame,
        model=texture_model,
    )

    return {
        "sky_mask": sky_mask, 
        "sky_bounding_box": sky_bounding_box,
        "texture_descriptor": texture_descriptor,
    }


def get_mask_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of the non-zero regions in the mask.

    Args:
        mask (np.ndarray): Binary mask where non-zero values indicate the area of interest.

    Returns:
        Tuple[int, int, int, int]: Bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    if mask is None or not np.any(mask):
        return 0, 0, mask.shape[1], mask.shape[0]  # Full frame if no mask

    y_indices, x_indices = np.where(mask)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return x_min, y_min, x_max + 1, y_max + 1


def process_video(
    video: cv2.VideoCapture,
    mask: Optional[np.ndarray],
    frame_step: int,
    video_frame_count: int,
    image_predictor: object,
    grounding_processor: object,
    grounding_model: object,
    box_threshold: float,
    text_threshold: float,
    texture_model: object,
) -> None:
    """
    Process the video frames with a specified frame step.

    Args:
        video (cv2.VideoCapture): Video capture object.
        mask (Optional[np.ndarray]): Binary mask to apply to the video frames.
        frame_step (int): Step size for processing frames.
        video_frame_count (int): Total number of frames in the video.
        image_predictor (Optional[object]): SAM2 image predictor model.
        grounding_processor (Optional[object]): Grounding DINO processor.
        grounding_model (Optional[object]): Grounding DINO model.
        box_threshold (float): Threshold for sky box detection.
        text_threshold (float): Threshold for sky text detection.
        texture_model (object): Texture descriptor model.

    Raises:
        ValueError: If frame_step is not a positive integer.
        ValueError: If video_frame_count is not a positive integer.
    """
    if frame_step <= 0:
        raise ValueError("❌ Frame step must be a positive integer.")
    if video_frame_count <= 0:
        raise ValueError("❌ Video frame count must be a positive integer.")

    # Get mask bounds to crop the area of interest
    bounds = get_mask_bounds(mask) if mask is not None else None

    n_frames_to_process = video_frame_count // frame_step
    bar = tqdm(
        total=n_frames_to_process, desc="⌛ Processing video frames...", unit="frame"
    )
    frame_count = 0
    sky_mask = None
    sky_bounding_box = None
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            frame_dict = process_frame(
                frame=frame,
                mask=mask,
                bounds=bounds,
                sky_mask=sky_mask,
                sky_bounding_box=sky_bounding_box,
                image_predictor=image_predictor if sky_mask is None else None,
                grounding_processor=grounding_processor if sky_mask is None else None,
                grounding_model=grounding_model if sky_mask is None else None,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                texture_model=texture_model,
            )

            if sky_mask is None:
                sky_mask = frame_dict.get("sky_mask")
            if sky_bounding_box is None:
                sky_bounding_box = frame_dict.get("sky_bounding_box")
                
            texture_descriptor = frame_dict.get("texture_descriptor")
            
            bar.update(1)

        frame_count += 1


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline.")

    parser.add_argument(
        "--video-path",
        "-vp",
        type=str,
        required=True,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--mask-path",
        "-mp",
        type=str,
        default=None,
        help="Path to the mask file, masking and cropping the area of interest in the video. If not provided, the whole video will be processed.",
    )
    parser.add_argument(
        "--frame-rate",
        "-fr",
        type=float,
        default=1/3,
        help=f"Frame rate for processing the video (default: 1/3).",
    )
    parser.add_argument(
        "--sam2-type",
        "-sam2",
        type=str,
        default="large",
        choices=["tiny", "small", "base", "large"],
        help="Type of SAM2 model to use (default: large).",
    )
    parser.add_argument(
        "--gdino-type",
        "-gdino",
        type=str,
        default="tiny",
        choices=["tiny", "base"],
        help="Type of Grounding DINO model to use (default: tiny).",
    )
    parser.add_argument(
        "--box-threshold",
        "-bt",
        type=float,
        default=0.35,
        help="Threshold for sky box detection (default: 0.35).",
    )
    parser.add_argument(
        "--text-threshold",
        "-tt",
        type=float,
        default=0.35,
        help="Threshold for sky text detection (default: 0.35).",
    )

    return parser.parse_args()


def main() -> None:
    # Get command line arguments
    args = parse_args()
    video_path = args.video_path
    mask_path = args.mask_path
    frame_rate = args.frame_rate
    sam2_type = args.sam2_type
    gdino_type = args.gdino_type
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    if frame_rate <= 0:
        raise ValueError("❌ Frame rate must be a positive number greater than zero.")
    if box_threshold < 0 or box_threshold > 1:
        raise ValueError("❌ Box threshold must be between 0 and 1.")
    if text_threshold < 0 or text_threshold > 1:
        raise ValueError("❌ Text threshold must be between 0 and 1.")
    
    ### TODO
    from src.pipeline.texture_descriptor import get_sky_finder_texture_descriptors, plot_sky_finder_texture_descriptors
    sky_finder_texture_descriptors = get_sky_finder_texture_descriptors()
    plot_sky_finder_texture_descriptors(sky_finder_texture_descriptors)
    ### TODO

    print(f"▶️  Running pipeline with video path {os.path.abspath(video_path)}.")
    if mask_path:
        print(
            f"➡️  Using mask at {os.path.abspath(mask_path)} to crop the area of interest."
        )

    # Get video
    video, video_frame_rate, video_frame_count = get_video(video_path)
    if frame_rate < 1:
        frame_period = 1 / frame_rate
        print(
            f"➡️  Processing video at 1 frame every {frame_period:.1f} seconds (Original: {video_frame_rate:.1f} FPS)."
        )
    else:
        print(
            f"➡️  Processing video at {frame_rate:.1f} FPS (original: {video_frame_rate:.1f} FPS)."
        )

    # Get mask if provided
    mask = None
    if mask_path:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"❌ Mask file not found at {os.path.abspath(mask_path)}."
            )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(
                f"❌ Could not read mask file at {os.path.abspath(mask_path)}. Ensure it is a valid image file."
            )
        mask = mask > (255 / 2)

    # Get sky segmentation models
    image_predictor, grounding_processor, grounding_model = get_sky_segmentation_models(
        sam2_type=sam2_type,
        gdino_type=gdino_type,
    )

    # Get texture model
    texture_model = get_texture_model()

    # Process video frames
    frame_step = int(video_frame_rate / frame_rate)
    process_video(
        video=video,
        mask=mask,
        frame_step=frame_step,
        video_frame_count=video_frame_count,
        image_predictor=image_predictor,
        grounding_processor=grounding_processor,
        grounding_model=grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        texture_model=texture_model,
    )


if __name__ == "__main__":
    main()
