import os
import sys
import wget
import glob
import json
import random
import shutil
import zipfile
import argparse
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.file import get_paths_recursive
from src.utils.random import set_seed

from src.config import (
    SEED,
    SKY_FINDER_PATH,
    SKY_FINDER_VAL_SPLIT,
    SKY_FINDER_CAMERA_IDS,
    SKY_FINDER_TRAIN_SPLIT,
    SKY_FINDER_IMAGES_PATH,
    SKY_FINDER_ARCHIVES_PATH,
    SKY_FINDER_EXTRACTED_PATH,
    KSY_FINDER_CATEGORY_MAPPING_FILE_PATH,
)


def download_archive(
    camera_id: str,
) -> bool:
    """
    Download a single camera archive.

    Args:
        camera_id (str): The camera ID to download.

    Returns:
        bool: True if download succeeded, False if failed, None if already exists.
    """
    archive_path = os.path.join(SKY_FINDER_ARCHIVES_PATH, f"{camera_id}.zip")
    if os.path.exists(archive_path):
        return None

    try:
        url = f"https://cs.valdosta.edu/~rpmihail/skyfinder/images/{camera_id}.zip"
        wget.download(url, out=archive_path, bar=None)
        return True
    except Exception as e:
        return False


def download_archives(
    max_workers: int,
    force: bool,
) -> None:
    """
    Download the sky finder archives from the specified URL.

    Args:
        max_workers (int): The maximum number of concurrent download workers.
        force (bool): Whether to force re-download the archives.
    """
    print("▶️  Downloading archives from https://cs.valdosta.edu/~rpmihail/skyfinder/images/...")

    # Delete existing archives if force is True
    if force:
        print("⚠️  Force re-download enabled. Deleting existing archives...")
        try:
            archive_paths = get_paths_recursive(
                folder_path=SKY_FINDER_ARCHIVES_PATH,
                match_pattern="*.zip",
                path_type="f",
                recursive=False,
            )
            for archive in archive_paths:
                os.remove(archive)
            print(f"🗑️  Removed {len(archive_paths)} existing archive(s).")
        except Exception as e:
            print(f"⚠️  Could not remove existing archives: {e}")

    # Ensure download directory exists
    os.makedirs(SKY_FINDER_ARCHIVES_PATH, exist_ok=True)

    # Get list of camera IDs to download
    camera_ids_to_download = []
    for camera_id in SKY_FINDER_CAMERA_IDS:
        archive_path = os.path.join(SKY_FINDER_ARCHIVES_PATH, f"{camera_id}.zip")
        if not os.path.exists(archive_path):
            camera_ids_to_download.append(camera_id)

    if not camera_ids_to_download:
        print("✅ All archives already downloaded.")
        return

    # Download archives concurrently
    print(f"➡️  Found {len(camera_ids_to_download)} archive(s) to download.")
    successful_downloads = 0
    failed_downloads = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_camera = {
            executor.submit(download_archive, camera_id): camera_id
            for camera_id in camera_ids_to_download
        }

        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_camera):
            camera_id = future_to_camera[future]
            try:
                result = future.result()
                if result is True:
                    successful_downloads += 1
                    print(f"✅ Successfully downloaded {camera_id}.zip")
                elif result is None:
                    print(f"ℹ️  Archive {camera_id}.zip already exists")
                else:
                    failed_downloads += 1
                    print(f"❌ Failed to download {camera_id}.zip")
            except Exception as e:
                failed_downloads += 1
                print(f"❌ Failed to download {camera_id}.zip: {e}")

    print(f"✅ Archive download completed. Success: {successful_downloads}, Failed: {failed_downloads}")


def extract_archives(
    force: bool,
) -> None:
    """
    Extract the downloaded archives.

    Args:
        force (bool): Whether to force re-extraction of archives.
    """
    print("▶️  Extracting archives...")

    # Delete existing extracted archives if force is True
    if force:
        print("⚠️  Force re-extraction enabled. Deleting existing extracted archives...")
        try:
            extracted_dir_paths = get_paths_recursive(
                folder_path=SKY_FINDER_EXTRACTED_PATH,
                match_pattern="*",
                path_type="d",
                recursive=False,
            )
            for extracted_dir_path in extracted_dir_paths:
                shutil.rmtree(extracted_dir_path, ignore_errors=True)
            print(f"🗑️  Removed {len(extracted_dir_paths)} existing extracted director(ies).")
        except Exception as e:
            print(f"⚠️  Could not remove existing extracted data: {e}")

    # Ensure extraction directory exists
    os.makedirs(SKY_FINDER_EXTRACTED_PATH, exist_ok=True)

    # Get list of archives to extract
    archive_file_paths = get_paths_recursive(
        folder_path=SKY_FINDER_ARCHIVES_PATH,
        match_pattern="*.zip",
        path_type="f",
        recursive=False,
    )
    archive_file_paths_to_extract = []
    for archive_file_path in archive_file_paths:
        camera_id = os.path.basename(archive_file_path).replace(".zip", "")
        extracted_camera_dir = os.path.join(SKY_FINDER_EXTRACTED_PATH, camera_id)
        if not os.path.exists(extracted_camera_dir):
            archive_file_paths_to_extract.append(archive_file_path)
    
    if not archive_file_paths_to_extract:
        print("✅ No archives to extract.")
        return

    # Extract archives sequentially
    print(f"➡️  Found {len(archive_file_paths_to_extract)} archive(s) to extract.")
    successful_extractions = 0
    failed_extractions = 0

    for archive_file_path in tqdm(archive_file_paths_to_extract, desc="⏳ Extracting archives..."):
        camera_id = os.path.basename(archive_file_path).replace(".zip", "")
        extracted_camera_dir = os.path.join(SKY_FINDER_EXTRACTED_PATH, camera_id)
        os.makedirs(extracted_camera_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
                zip_ref.extractall(extracted_camera_dir)
            successful_extractions += 1
        except Exception as e:
            failed_extractions += 1
            print(f"❌ Failed to extract {camera_id}.zip: {e}")
            continue

        # Move images to root directory
        try:
            image_file_paths = glob.glob(
                os.path.join(extracted_camera_dir, "**/*.jpg"), recursive=True
            )
            moved_images = 0
            for image_file_path in image_file_paths:
                # Skip if already in root directory
                if os.path.dirname(image_file_path) == extracted_camera_dir:
                    continue

                filename = os.path.basename(image_file_path)
                target_path = os.path.join(extracted_camera_dir, filename)

                try:
                    os.rename(image_file_path, target_path)
                    moved_images += 1
                except Exception as e:
                    print(f"❌ Failed to move {image_file_path} to {target_path}: {e}")
                    continue

            # Remove empty directories
            for root, dirs, files in os.walk(extracted_camera_dir, topdown=False):
                for dir_name in dirs:
                    try:
                        dir_path = os.path.join(root, dir_name)
                        os.rmdir(dir_path)
                    except OSError:
                        pass

        except Exception as e:
            print(f"⚠️  Could not reorganize images for {camera_id}: {e}")

    print(f"✅ Archive extraction completed. Success: {successful_extractions}, Failed: {failed_extractions}")


def classify_extracted_images(
    force: bool,
) -> None:
    """
    Classify the extracted images into categories based on the category mapping.

    Args:
        force (bool): Whether to force re-classification of images.
    """
    print("▶️  Classifying extracted images...")

    if force:
        print("⚠️  Force re-classification enabled. Deleting existing classified images...")
        try:
            category_dir_paths = get_paths_recursive(
                folder_path=SKY_FINDER_IMAGES_PATH,
                match_pattern="*",
                path_type="d",
                recursive=False,
            )
            for category_dir_path in category_dir_paths:
                if os.path.isdir(category_dir_path):
                    shutil.rmtree(category_dir_path, ignore_errors=True)
            print(f"🗑️  Removed {len(category_dir_paths)} existing category director(ies).")
        except Exception as e:
            print(f"⚠️  Could not remove existing classified images: {e}")

    # Load the category mapping
    try:
        with open(KSY_FINDER_CATEGORY_MAPPING_FILE_PATH, "r") as f:
            category_mapping = json.load(f)
        print(f"✅ Successfully loaded category mapping with {len(category_mapping)} entries.")
    except Exception as e:
        print(f"❌ Failed to load category mapping: {e}")
        return

    # Create category directories
    os.makedirs(SKY_FINDER_IMAGES_PATH, exist_ok=True)
    categories = set(category_mapping.values())
    for category in categories:
        os.makedirs(os.path.join(SKY_FINDER_IMAGES_PATH, category), exist_ok=True)
    print(f"📁 Created {len(categories)} category directories.")

    # Get camera directories
    camera_dirs = glob.glob(os.path.join(SKY_FINDER_EXTRACTED_PATH, "*"))
    camera_dirs = [d for d in camera_dirs if os.path.isdir(d)]
    if not camera_dirs:
        print("❌ No extracted camera directories found.")
        return
    print(f"➡️  Found {len(camera_dirs)} camera directories to process.")

    # Classify images
    total_classified = 0
    skipped_images = 0
    failed_classifications = 0

    for camera_dir in tqdm(camera_dirs, desc="⏳ Classifying images..."):
        camera_id = os.path.basename(camera_dir)
        image_file_paths = glob.glob(os.path.join(camera_dir, "*.jpg"))
        if not image_file_paths:
            print(f"⚠️  No images found in {camera_dir}")
            continue

        for image_file_path in image_file_paths:
            image_filename = os.path.basename(image_file_path)
            mapping_key = f"{camera_id}/{image_filename}"
            
            if mapping_key not in category_mapping:
                skipped_images += 1
                continue

            category = category_mapping[mapping_key]
            dest_dir = os.path.join(SKY_FINDER_IMAGES_PATH, category, camera_id)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, image_filename)

            # Skip if already classified
            if os.path.exists(dest_path):
                continue

            try:
                shutil.copy2(image_file_path, dest_path)
                total_classified += 1
            except Exception as e:
                failed_classifications += 1
                print(f"❌ Failed to copy {image_file_path} to {dest_path}: {e}")
                continue

    print(f"✅ Image classification completed. Classified: {total_classified}, Skipped: {skipped_images}, Failed: {failed_classifications}")


def split_classified_images(force: bool) -> None:
    """
    Split the classified images into training, validation, and test sets.

    Args:
        force (bool): Whether to force re-splitting of images.
    """
    print("▶️  Splitting classified images into train, val, and test sets...")

    if force:
        print("⚠️  Force re-splitting enabled. Deleting existing split directories...")
        directory_names = ["train", "val", "test"]
        removed_dirs = 0
        for directory_name in directory_names:
            directory_path = f"{SKY_FINDER_PATH}{directory_name}/"
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path, ignore_errors=True)
                removed_dirs += 1
        print(f"🗑️  Removed {removed_dirs} existing split director(ies).")

    # Create the split directories
    directory_names = ["train", "val", "test"]
    for directory_name in directory_names:
        directory_path = f"{SKY_FINDER_PATH}{directory_name}/"
        os.makedirs(directory_path, exist_ok=True)

    # Get all images and calculate split indices
    image_file_paths = get_paths_recursive(
        folder_path=SKY_FINDER_IMAGES_PATH,
        match_pattern="*.jpg",
        path_type="f",
        recursive=True,
    )

    if not image_file_paths:
        print("❌ No classified images found to split.")
        return

    set_seed(SEED)
    n_images = len(image_file_paths)
    indices = list(range(n_images))
    random_indices = indices.copy()
    random.shuffle(random_indices)

    train_indices = set(random_indices[: int(n_images * SKY_FINDER_TRAIN_SPLIT)])
    val_indices = set(random_indices[
        int(n_images * SKY_FINDER_TRAIN_SPLIT) : int(
            n_images * (SKY_FINDER_TRAIN_SPLIT + SKY_FINDER_VAL_SPLIT)
        )
    ])

    print(f"📊 Splitting {n_images} images: {len(train_indices)} train, {len(val_indices)} val, {n_images - len(train_indices) - len(val_indices)} test")

    # Split images into directories
    train_count = val_count = test_count = 0
    failed_copies = 0

    for i, image_file_path in tqdm(enumerate(image_file_paths), desc="⏳ Splitting images...", total=n_images):
        if i in train_indices:
            directory_name = "train"
            train_count += 1
        elif i in val_indices:
            directory_name = "val"
            val_count += 1
        else:
            directory_name = "test"
            test_count += 1

        local_image_file_path = "/".join(image_file_path.split("/")[-3:])
        new_image_file_path = f"{SKY_FINDER_PATH}{directory_name}/{local_image_file_path}"
        os.makedirs(os.path.dirname(new_image_file_path), exist_ok=True)

        if os.path.exists(new_image_file_path):
            continue

        try:
            shutil.copy2(image_file_path, new_image_file_path)
        except Exception as e:
            failed_copies += 1
            print(f"❌ Failed to copy {image_file_path} to {new_image_file_path}: {e}")
            continue

    print(f"✅ Image splitting completed. Train: {train_count}, Val: {val_count}, Test: {test_count}, Failed: {failed_copies}")


def remove_data() -> None:
    """
    Remove the downloaded archives and extracted data.
    """
    print("▶️  Removing temporary data...")
    
    removed_items = 0

    # Remove archives
    try:
        archive_paths = get_paths_recursive(
            folder_path=SKY_FINDER_ARCHIVES_PATH,
            match_pattern="*.zip",
            path_type="f",
            recursive=False,
        )
        for archive_path in archive_paths:
            os.remove(archive_path)
            removed_items += 1
        if os.path.exists(SKY_FINDER_ARCHIVES_PATH):
            os.rmdir(SKY_FINDER_ARCHIVES_PATH)
        print(f"🗑️  Removed {len(archive_paths)} archive(s)")
    except Exception as e:
        print(f"⚠️  Could not remove archives: {e}")

    # Remove extracted data
    try:
        extracted_dir_paths = get_paths_recursive(
            folder_path=SKY_FINDER_EXTRACTED_PATH,
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        for extracted_dir_path in extracted_dir_paths:
            shutil.rmtree(extracted_dir_path, ignore_errors=True)
            removed_items += 1
        if os.path.exists(SKY_FINDER_EXTRACTED_PATH):
            os.rmdir(SKY_FINDER_EXTRACTED_PATH)
        print(f"🗑️  Removed {len(extracted_dir_paths)} extracted director(ies)")
    except Exception as e:
        print(f"⚠️  Could not remove extracted data: {e}")

    # Remove classified images
    try:
        if os.path.exists(SKY_FINDER_IMAGES_PATH):
            shutil.rmtree(SKY_FINDER_IMAGES_PATH, ignore_errors=True)
            removed_items += 1
            print("🗑️  Removed classified images directory")
    except Exception as e:
        print(f"⚠️  Could not remove classified images: {e}")

    print(f"✅ Cleanup completed. Total items removed: {removed_items}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate sky finder dataset by downloading, extracting, classifying, and splitting images.")

    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=3,
        help="Number of concurrent download workers (default: 3)",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-download, re-extraction, re-classification, and re-splitting of data",
    )

    parser.add_argument(
        "-r",
        "--remove-data",
        action="store_true",
        help="Remove temporary archives and extracted data after processing (keeps final split data)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to generate sky finder dataset.
    """
    args = parse_args()

    print("▶️  Starting sky finder dataset generation...")
    print(f"📋 Configuration:")
    print(f"   • Max workers: {args.max_workers}")
    print(f"   • Force re-processing: {args.force}")
    print(f"   • Remove temporary data: {args.remove_data}")
    print(f"   • Archives path: {os.path.abspath(SKY_FINDER_ARCHIVES_PATH)}")
    print(f"   • Extracted path: {os.path.abspath(SKY_FINDER_EXTRACTED_PATH)}")
    print(f"   • Classified images path: {os.path.abspath(SKY_FINDER_IMAGES_PATH)}")
    print(f"   • Final dataset path: {os.path.abspath(SKY_FINDER_PATH)}")
    print(f"   • Category mapping: {os.path.abspath(KSY_FINDER_CATEGORY_MAPPING_FILE_PATH)}")
    print(f"   • Train split: {SKY_FINDER_TRAIN_SPLIT}")
    print(f"   • Validation split: {SKY_FINDER_VAL_SPLIT}")
    print(f"   • Test split: {1 - SKY_FINDER_TRAIN_SPLIT - SKY_FINDER_VAL_SPLIT}")
    print(f"   • Random seed: {SEED}")

    try:
        # Execute pipeline steps
        download_archives(max_workers=args.max_workers, force=args.force)
        extract_archives(force=args.force)
        classify_extracted_images(force=args.force)
        split_classified_images(force=args.force)

        # Clean up temporary data if requested
        if args.remove_data:
            remove_data()

        print("🎉 Sky finder dataset generation completed successfully!")

    except Exception as e:
        print(f"💥 Fatal error during dataset generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()