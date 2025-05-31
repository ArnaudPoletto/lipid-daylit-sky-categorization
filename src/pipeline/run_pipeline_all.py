import os
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (
    PROCESSED_VIDEOS_PATH,
)


def get_video_files(videos_path: str) -> list:
    """
    Get all video files from the specified directory.

    Args:
        videos_path (str): Path to the directory containing video files.

    Returns:
        list: List of video file paths.
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = []

    if not os.path.exists(videos_path):
        raise FileNotFoundError(
            f"❌ Videos directory not found at {os.path.abspath(videos_path)}"
        )

    for file in os.listdir(videos_path):
        if file.lower().endswith(video_extensions):
            video_files.append(os.path.join(videos_path, file))

    if not video_files:
        raise ValueError(f"❌ No video files found in {os.path.abspath(videos_path)}")

    video_files.sort()  # Sort for consistent ordering
    return video_files


def run_pipeline_for_video(video_path: str, mask_path: str = None, **kwargs) -> bool:
    """
    Run the pipeline for a single video.

    Args:
        video_path (str): Path to the video file.
        mask_path (str, optional): Path to the mask file.
        **kwargs: Additional arguments to pass to the pipeline.

    Returns:
        bool: True if successful, False otherwise.
    """
    cmd = [sys.executable, "run_pipeline.py", "-vp", video_path]

    if mask_path:
        cmd.extend(["-mp", mask_path])

    # Add other arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key.replace('_', '-')}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    try:
        print(f"🚀 Running pipeline for: {os.path.basename(video_path)}")
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Successfully processed: {os.path.basename(video_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {os.path.basename(video_path)}: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline for all videos in the processed videos directory."
    )

    parser.add_argument(
        "--mask-path",
        "-mp",
        type=str,
        default=None,
        help="Path to the mask file to use for all videos. If not provided, no mask will be applied.",
    )
    parser.add_argument(
        "--frame-rate",
        "-fr",
        type=float,
        default=1 / 3,
        help="Frame rate for processing the videos (default: 1/3).",
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
        "--continue-on-error",
        "-coe",
        action="store_true",
        help="Continue processing other videos if one fails.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the pipeline for all videos.
    """
    args = parse_args()

    print(
        f"▶️  Running pipeline for all videos in {os.path.abspath(PROCESSED_VIDEOS_PATH)}."
    )

    # Get all video files
    try:
        video_files = get_video_files(PROCESSED_VIDEOS_PATH)
        print(
            f"➡️  Found {len(video_files)} video(s) in {os.path.abspath(PROCESSED_VIDEOS_PATH)}."
        )
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    # Prepare arguments for pipeline
    pipeline_args = {
        "frame_rate": args.frame_rate,
        "sam2_type": args.sam2_type,
        "gdino_type": args.gdino_type,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }

    # Process all videos
    successful = 0
    failed = 0

    print(f"🔄 Starting batch processing of {len(video_files)} video(s)...")

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing {os.path.basename(video_path)}.")

        success = run_pipeline_for_video(
            video_path=video_path, mask_path=args.mask_path, **pipeline_args
        )

        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print(
                    f"❌ Stopping due to error. Use --continue-on-error to skip failed videos."
                )
                break

    # Summary
    print(f"📊 Processing Summary:")
    print(f"\t✅ Successful: {successful}")
    print(f"\t❌ Failed: {failed}")
    print(f"\t📁 Total: {len(video_files)}")

    if failed == 0:
        print("🎉 All videos processed successfully!")
    elif successful > 0:
        print("⚠️  Some videos failed to process.")
    else:
        print("💥 No videos were processed successfully.")


if __name__ == "__main__":
    main()
