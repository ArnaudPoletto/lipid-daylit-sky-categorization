import os
import sys
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.unet import UNet
from src.pipeline.sky_image_descriptor import (
    get_sky_image_descriptor,
    get_sky_finder_descriptors,
    get_fitted_umap_reducer,
    plot_sky_finder_sky_image_descriptors,
)
from src.pipeline.sky_cover import get_sky_cover
from src.models.sky_class_net import SkyClassNet
from src.models.contrastive_net import ContrastiveNet
from src.pipeline.sky_segmentation import get_sky_mask
from src.pipeline.sky_classification import get_sky_class
from src.pipeline.sky_optical_flow import get_optical_flow
from src.pipeline.sky_cover import get_model as get_sky_cover_model
from src.pipeline.sky_image_descriptor import get_model as get_sky_image_model
from src.pipeline.sky_classification import get_model as get_sky_class_model
from src.pipeline.sky_segmentation import get_models as get_sky_segmentation_models
from src.config import (
    GENERATED_PIPELINE_PATH,
)


def get_video(video_path: str) -> Tuple[cv2.VideoCapture, float, int]:
    """
    Get a video capture object for the specified video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Tuple[cv2.VideoCapture, float, int]: Video capture object, frame rate, and frame count.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video file format is not supported.
        IOError: If the video file cannot be opened.
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


def get_mask(mask_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Get a binary mask from the specified mask file path.

    Args:
        mask_path (Optional[str]): Path to the mask file. If None, no mask is applied.

    Returns:
        Optional[np.ndarray]: Binary mask as a numpy array, or None if no mask is provided.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If the mask file cannot be read.
    """
    if mask_path is None:
        return None

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"❌ Mask file not found at {os.path.abspath(mask_path)}."
        )

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(
            f"❌ Could not read mask file at {os.path.abspath(mask_path)}. Ensure it is a valid image file."
        )

    return mask > (255 / 2)


def get_mask_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of the non-zero regions in the mask.

    Args:
        mask (np.ndarray): Binary mask where non-zero values indicate the area of interest.

    Returns:
        Tuple[int, int, int, int]: Bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    if mask is None or not np.any(mask):
        return 0, 0, mask.shape[1], mask.shape[0]

    y_indices, x_indices = np.where(mask)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return x_min, y_min, x_max + 1, y_max + 1


def process_frame(
    frame: np.ndarray,
    previous_frame: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    bounds: Optional[Tuple[int, int, int, int]],
    sky_mask: Optional[np.ndarray],
    sky_bounding_box: Optional[Tuple[int, int, int, int]],
    image_predictor: Optional[object],
    grounding_processor: Optional[object],
    grounding_model: Optional[object],
    box_threshold: float,
    text_threshold: float,
    sky_image_model: ContrastiveNet,
    sky_class_model: SkyClassNet,
    sky_cover_model: UNet,
) -> Dict[str, Any]:
    """
    Process a single video frame to extract sky-related features.

    Args:
        frame (np.ndarray): The video frame to process.
        previous_frame (Optional[np.ndarray]): The previous video frame for optical flow computation.
        mask (Optional[np.ndarray]): Binary mask to apply to the frame.
        bounds (Optional[Tuple[int, int, int, int]]): Bounding box coordinates to crop the frame.
        sky_mask (Optional[np.ndarray]): Precomputed sky mask for the frame.
        sky_bounding_box (Optional[Tuple[int, int, int, int]]): Bounding box for the sky mask.
        image_predictor (Optional[object]): SAM2 image predictor model.
        grounding_processor (Optional[object]): Grounding DINO processor.
        grounding_model (Optional[object]): Grounding DINO model.
        box_threshold (float): Threshold for sky box detection.
        text_threshold (float): Threshold for sky text detection.
        sky_image_model (ContrastiveNet): Sky image descriptor model.
        sky_class_model (SkyClassNet): Sky classification model.
        sky_cover_model (UNet): Sky cover model.

    Returns:
        Dict[str, Any]: Dictionary containing sky mask, bounding box, and extracted features.
    """
    # Apply the first mask if provided
    if mask is not None:
        frame *= mask[:, :, np.newaxis]
        frame = frame[bounds[1] : bounds[3], bounds[0] : bounds[2]]
        if previous_frame is not None:
            previous_frame *= mask[:, :, np.newaxis]
            previous_frame = previous_frame[bounds[1] : bounds[3], bounds[0] : bounds[2]]

    # Get sky mask if models are provided and not already computed
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
        if previous_frame is not None:
            previous_frame = previous_frame[y_min:y_max, x_min:x_max]
    
    if sky_mask is not None:
        frame *= sky_mask[:, :, np.newaxis]
        if previous_frame is not None:
            previous_frame *= sky_mask[:, :, np.newaxis]

    # Apply inpainting to fill the masked area
    inpainted_frame = frame
    if sky_mask is not None:
        inpaint_mask = (~(sky_mask.astype(np.bool_))).astype(np.uint8) * 255
        inpainted_frame = cv2.inpaint(frame, inpaint_mask, 3, cv2.INPAINT_TELEA)

    # Get sky image descriptor
    sky_image_descriptor = get_sky_image_descriptor(
        frame=inpainted_frame,
        model=sky_image_model,
    )
    normalized_sky_image_descriptor = sky_image_descriptor / np.linalg.norm(sky_image_descriptor)

    # Get sky class prediction
    sky_class = get_sky_class(
        sky_image_descriptor=sky_image_descriptor,
        model=sky_class_model,
    )

    # Get sky cover prediction
    sky_cover = get_sky_cover(
        frame=frame,
        model=sky_cover_model,
    )

    # Get optical flow magnitude if previous frame is available
    optical_flow_magnitude = None
    if previous_frame is not None:
        optical_flow = get_optical_flow(
            frame=frame,
            previous_frame=previous_frame,
            mask=sky_mask,
        )
        optical_flow_magnitude = np.linalg.norm(optical_flow, axis=2)
        optical_flow_magnitude_values = optical_flow_magnitude.flatten()[sky_mask.flatten() > 0]
        
        if optical_flow_magnitude_values.size > 0:
            optical_flow_magnitude = float(np.mean(optical_flow_magnitude_values))

    return {
        "sky_mask": sky_mask,
        "sky_bounding_box": sky_bounding_box,
        "sky_image_descriptor": normalized_sky_image_descriptor.tolist(),
        "sky_class": int(sky_class),
        "sky_cover": float(sky_cover),
        "optical_flow_magnitude": optical_flow_magnitude,
    }


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
    sky_image_model: ContrastiveNet,
    sky_class_model: SkyClassNet,
    sky_cover_model: UNet,
) -> Dict[str, Any]:
    """
    Process the video frames with a specified frame step.

    Args:
        video (cv2.VideoCapture): Video capture object.
        mask (Optional[np.ndarray]): Binary mask to apply to the video frames.
        frame_step (int): Step size for processing frames.
        video_frame_count (int): Total number of frames in the video.
        image_predictor (object): SAM2 image predictor model.
        grounding_processor (object): Grounding DINO processor.
        grounding_model (object): Grounding DINO model.
        box_threshold (float): Threshold for sky box detection.
        text_threshold (float): Threshold for sky text detection.
        sky_image_model (ContrastiveNet): Sky image descriptor model.
        sky_class_model (SkyClassNet): Sky classification model.
        sky_cover_model (UNet): Sky cover model.

    Returns:
        Dict[str, Any]: Dictionary containing the processed results.

    Raises:
        ValueError: If frame_step or video_frame_count is not a positive integer.
    """
    if frame_step <= 0:
        raise ValueError("❌ Frame step must be a positive integer.")
    if video_frame_count <= 0:
        raise ValueError("❌ Video frame count must be a positive integer.")

    # Get mask bounds to crop the area of interest
    bounds = get_mask_bounds(mask) if mask is not None else None

    # Process video frames sequentially
    n_frames_to_process = video_frame_count // frame_step
    sky_mask = None
    sky_bounding_box = None
    video_dict = {}
    previous_frame = None
    
    with tqdm(total=n_frames_to_process, desc="⏳ Processing video frames", unit="frame") as pbar:
        frame_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # BGR to RGB conversion
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = frame.copy()

            # Process current frame
            frame_dict = process_frame(
                frame=frame,
                previous_frame=previous_frame,
                mask=mask,
                bounds=bounds,
                sky_mask=sky_mask,
                sky_bounding_box=sky_bounding_box,
                image_predictor=image_predictor if sky_mask is None else None,
                grounding_processor=grounding_processor if sky_mask is None else None,
                grounding_model=grounding_model if sky_mask is None else None,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                sky_image_model=sky_image_model,
                sky_class_model=sky_class_model,
                sky_cover_model=sky_cover_model,
            )

            # Update sky mask and bounding box if not already set
            if sky_mask is None:
                sky_mask = frame_dict.get("sky_mask")
            if sky_bounding_box is None:
                sky_bounding_box = frame_dict.get("sky_bounding_box")

            # Store frame results
            video_dict[frame_count] = {
                "sky_image_descriptor": frame_dict.get("sky_image_descriptor"),
                "sky_class": frame_dict.get("sky_class"),
                "sky_cover": frame_dict.get("sky_cover"),
                "optical_flow_magnitude": frame_dict.get("optical_flow_magnitude"),
            }

            # Update to next frame
            previous_frame = original_frame
            pbar.update(1)
            frame_count += frame_step
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    video.release()

    # Compute global video statistics
    sky_image_descriptors = np.array([
        video_dict[frame]["sky_image_descriptor"] 
        for frame in video_dict
    ])
    mean_sky_image_descriptor = np.mean(sky_image_descriptors, axis=0)
    
    sky_covers = [video_dict[frame]["sky_cover"] for frame in video_dict]
    mean_sky_cover = float(np.mean(sky_covers))
    
    sky_classes = [video_dict[frame]["sky_class"] for frame in video_dict]
    majority_sky_class = int(np.bincount(sky_classes).argmax())

    optical_flow_magnitudes = [video_dict[frame]["optical_flow_magnitude"] for frame in video_dict]
    optical_flow_magnitudes = [m for m in optical_flow_magnitudes if m is not None]
    mean_optical_flow_magnitude = float(np.mean(optical_flow_magnitudes))
    
    # Add global statistics to results
    video_dict["mean_sky_image_descriptor"] = mean_sky_image_descriptor.tolist()
    video_dict["mean_sky_cover"] = mean_sky_cover
    video_dict["majority_sky_class"] = majority_sky_class
    video_dict["mean_optical_flow_magnitude"] = mean_optical_flow_magnitude

    return video_dict


def save_results(
    video_dict: Dict[str, Any],
    video_path: str,
) -> str:
    """
    Save the processed video results to a JSON file.

    Args:
        video_dict (Dict[str, Any]): Dictionary containing the processed results.
        video_path (str): Path to the original video file.

    Returns:
        str: Path to the saved results file.
    """
    os.makedirs(GENERATED_PIPELINE_PATH, exist_ok=True)
    
    video_name = os.path.basename(video_path).replace(".", "_")
    output_path = os.path.join(GENERATED_PIPELINE_PATH, f"{video_name}_processed.json")
    
    with open(output_path, "w") as f:
        json.dump(video_dict, f, indent=4)
    
    return output_path


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the sky analysis pipeline on a video file.")

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
        help="Path to the mask file for cropping the area of interest. If not provided, the whole video will be processed.",
    )
    
    parser.add_argument(
        "--frame-rate",
        "-fr",
        type=float,
        default=1/3,
        help="Frame rate for processing the video (default: 1/3 FPS).",
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
    
    parser.add_argument(
        "--show-plots",
        "-sp",
        action="store_true",
        help="Show plots of the processed video results.",
    )
    
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-processing even if results already exist.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the sky analysis pipeline.
    """
    args = parse_args()
    
    print("▶️  Starting sky analysis pipeline...")
    print(f"📋 Configuration:")
    print(f"   • Video path: {os.path.abspath(args.video_path)}")
    if args.mask_path:
        print(f"   • Mask path: {os.path.abspath(args.mask_path)}")
    print(f"   • Frame rate: {args.frame_rate} FPS")
    print(f"   • SAM2 model: {args.sam2_type}")
    print(f"   • Grounding DINO model: {args.gdino_type}")
    print(f"   • Box threshold: {args.box_threshold}")
    print(f"   • Text threshold: {args.text_threshold}")
    print(f"   • Show plots: {args.show_plots}")
    print(f"   • Force re-process: {args.force}")

    # Validate arguments
    if args.frame_rate <= 0:
        raise ValueError("❌ Frame rate must be a positive number greater than zero.")
    if not 0 <= args.box_threshold <= 1:
        raise ValueError("❌ Box threshold must be between 0 and 1.")
    if not 0 <= args.text_threshold <= 1:
        raise ValueError("❌ Text threshold must be between 0 and 1.")

    try:
        # Load video and mask
        print("▶️  Loading video...")
        video, video_frame_rate, video_frame_count = get_video(args.video_path)
        mask = get_mask(args.mask_path) if args.mask_path else None
        
        # Display frame processing information
        if args.frame_rate < 1:
            frame_period = 1 / args.frame_rate
            print(f"📊 Processing 1 frame every {frame_period:.1f} seconds (Original: {video_frame_rate:.1f} FPS)")
        else:
            print(f"📊 Processing at {args.frame_rate:.1f} FPS (Original: {video_frame_rate:.1f} FPS)")
        print(f"📊 Total frames: {video_frame_count}, Frames to process: {video_frame_count // int(video_frame_rate / args.frame_rate)}")

        # Check if results already exist
        video_name = os.path.basename(args.video_path).replace(".", "_")
        output_path = os.path.join(GENERATED_PIPELINE_PATH, f"{video_name}_processed.json")
        
        if os.path.exists(output_path) and not args.force:
            print(f"✅ Processed results already exist at {os.path.abspath(output_path)}")
            print("   Use --force to re-process.")
            
            # Load existing results
            with open(output_path, "r") as f:
                video_dict = json.load(f)
        else:
            # Load models
            print("▶️  Loading models...")
            image_predictor, grounding_processor, grounding_model = get_sky_segmentation_models(
                sam2_type=args.sam2_type,
                gdino_type=args.gdino_type,
            )
            sky_image_model = get_sky_image_model()
            sky_class_model = get_sky_class_model()
            sky_cover_model = get_sky_cover_model()
            print("✅ Models loaded successfully.")

            # Process video
            frame_step = int(video_frame_rate / args.frame_rate)
            video_dict = process_video(
                video=video,
                mask=mask,
                frame_step=frame_step,
                video_frame_count=video_frame_count,
                image_predictor=image_predictor,
                grounding_processor=grounding_processor,
                grounding_model=grounding_model,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                sky_image_model=sky_image_model,
                sky_class_model=sky_class_model,
                sky_cover_model=sky_cover_model,
            )

            # Save results
            output_path = save_results(video_dict, args.video_path)
            print(f"✅ Results saved to {os.path.abspath(output_path)}")

        # Show plots if requested
        if args.show_plots:
            print("▶️  Generating visualization...")
            
            # Get mean sky image descriptor
            mean_sky_image_descriptor = video_dict.get("mean_sky_image_descriptor")
            if mean_sky_image_descriptor is None:
                raise ValueError("❌ Mean sky image descriptor not found in processed results.")
            mean_sky_image_descriptor = np.array(mean_sky_image_descriptor)

            # Load Sky Finder descriptors and fit UMAP
            sky_finder_descriptors, cloud_coverages, sky_classes, image_paths = get_sky_finder_descriptors()
            fitted_umap_reducer = get_fitted_umap_reducer(
                sky_finder_sky_image_descriptors=sky_finder_descriptors
            )
            
            # Plot with video descriptor
            oos_sky_image_descriptors = np.array([mean_sky_image_descriptor])
            plot_sky_finder_sky_image_descriptors(
                fitted_umap=fitted_umap_reducer,
                sky_finder_sky_image_descriptors=sky_finder_descriptors,
                oos_sky_image_descriptors=oos_sky_image_descriptors,
                oos_labels=["Video Mean"],
            )

        print("🎉 Sky analysis pipeline completed successfully!")

    except Exception as e:
        print(f"💥 Fatal error during pipeline execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()