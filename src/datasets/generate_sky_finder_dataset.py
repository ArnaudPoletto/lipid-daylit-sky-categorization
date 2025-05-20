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
    SKY_FINDER_TEST_SPLIT,
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
    print(
        "▶️  Downloading archives from https://cs.valdosta.edu/~rpmihail/skyfinder/images/..."
    )

    # Delete existing archives if force is True
    if force:
        print("⚠️  Force re-download enabled. Deleting existing archives...")
        archive_paths = get_paths_recursive(
            folder_path=SKY_FINDER_ARCHIVES_PATH,
            match_pattern="*.zip",
            path_type="f",
            recursive=False,
        )
        for archive in archive_paths:
            os.remove(archive)

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
                    print(f"✅ Successfully downloaded {camera_id}.zip.")
                elif result is None:
                    print(f"File {camera_id}.zip already exists.")
            except Exception as e:
                print(f"❌ Failed to download {camera_id}.zip: {e}.")

    print("✅ All archives downloaded.")


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
        extracted_dir_paths = get_paths_recursive(
            folder_path=SKY_FINDER_EXTRACTED_PATH,
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        for extracted_dir_path in extracted_dir_paths:
            shutil.rmtree(extracted_dir_path, ignore_errors=True)

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
    for archive_file_path in tqdm(archive_file_paths_to_extract, desc="⏳ Extracting archives..."):
        camera_id = os.path.basename(archive_file_path).replace(".zip", "")
        extracted_camera_dir = os.path.join(SKY_FINDER_EXTRACTED_PATH, camera_id)
        os.makedirs(extracted_camera_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
                zip_ref.extractall(extracted_camera_dir)
        except Exception as e:
            print(f"❌ Failed to extract {camera_id}.zip: {e}.")
            continue

        # Move images to root directory
        image_file_paths = glob.glob(
            os.path.join(extracted_camera_dir, "**/*.jpg"), recursive=True
        )
        for image_file_path in image_file_paths:
            # Skip if already in root directory
            if os.path.dirname(image_file_path) == extracted_camera_dir:
                continue

            filename = os.path.basename(image_file_path)
            target_path = os.path.join(extracted_camera_dir, filename)

            try:
                os.rename(image_file_path, target_path)
            except Exception as e:
                print(f"❌ Failed to move {image_file_path} to {target_path}: {e}.")
                continue

        # Remove empty directories
        for root, dirs, files in os.walk(extracted_camera_dir, topdown=False):
            for dir_name in dirs:
                try:
                    dir_path = os.path.join(root, dir_name)
                    os.rmdir(dir_path)
                except OSError:
                    pass

    print("✅ All archives extracted.")


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
        print(
            "⚠️  Force re-classification enabled. Deleting existing classified images..."
        )
        category_dir_paths = get_paths_recursive(
            folder_path=SKY_FINDER_IMAGES_PATH,
            match_pattern="*",
            path_type="d",
            recursive=False,
        )
        for category_dir_path in category_dir_paths:
            if os.path.isdir(category_dir_path):
                shutil.rmtree(category_dir_path, ignore_errors=True)

    # Load the category mapping
    try:
        with open(KSY_FINDER_CATEGORY_MAPPING_FILE_PATH, "r") as f:
            category_mapping = json.load(f)
        print(f"✅ Loaded category mapping with {len(category_mapping)} entries.")
    except Exception as e:
        print(f"❌ Failed to load category mapping: {e}")
        return

    # Create category directories
    os.makedirs(SKY_FINDER_IMAGES_PATH, exist_ok=True)
    categories = set(category_mapping.values())
    for category in categories:
        os.makedirs(os.path.join(SKY_FINDER_IMAGES_PATH, category), exist_ok=True)

    # Get camera directories
    camera_dirs = glob.glob(os.path.join(SKY_FINDER_EXTRACTED_PATH, "*"))
    camera_dirs = [d for d in camera_dirs if os.path.isdir(d)]
    if not camera_dirs:
        print("❌ No extracted camera directories found")
        return
    print(f"➡️  Found {len(camera_dirs)} camera directories to process.")

    # Classify images
    for camera_dir in tqdm(camera_dirs, desc="⏳ Classifying images..."):
        camera_id = os.path.basename(camera_dir)
        image_file_paths = glob.glob(os.path.join(camera_dir, "*.jpg"))
        if not image_file_paths:
            print(f"❌ No images found in {camera_dir}.")
            continue

        for image_file_path in image_file_paths:
            image_filename = os.path.basename(image_file_path)
            mapping_key = f"{camera_id}/{image_filename}"
            if mapping_key not in category_mapping:
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
            except Exception as e:
                print(f"❌ Failed to copy {image_file_path} to {dest_path}: {e}.")
                continue

    print("✅ All images classified.")


def split_classified_images(force: bool) -> None:
    """
    Split the classified images into training, validation, and test sets.

    Args:
        force (bool): Whether to force re-splitting of images.
    """
    print("▶️  Splitting classified images into train, val, and test sets...")

    if force:
        print(
            "⚠️  Force re-splitting enabled. Deleting existing split directories..."
        )
        directory_names = ["train", "val", "test"]
        for directory_name in directory_names:
            directory_path = f"{SKY_FINDER_PATH}{directory_name}/"
            shutil.rmtree(directory_path, ignore_errors=True)

    # Create the split directories
    directory_names = ["train", "val", "test"]
    for directory_name in directory_names:
        directory_path = f"{SKY_FINDER_PATH}{directory_name}/"
        os.makedirs(directory_path, exist_ok=True)

    # Get all images and move them to the appropriate split directory
    image_file_paths = get_paths_recursive(
        folder_path=SKY_FINDER_IMAGES_PATH,
        match_pattern="*.jpg",
        path_type="f",
        recursive=True,
    )

    set_seed(SEED)
    n_images = len(image_file_paths)
    indices = list(range(n_images))
    random_indices = indices.copy()
    random.shuffle(random_indices)

    train_indices = random_indices[: int(n_images * SKY_FINDER_TRAIN_SPLIT)]
    val_indices = random_indices[
        int(n_images * SKY_FINDER_TRAIN_SPLIT) : int(
            n_images * (SKY_FINDER_TRAIN_SPLIT + SKY_FINDER_VAL_SPLIT)
        )
    ]

    for i, image_file_path in tqdm(enumerate(image_file_paths), desc="⏳ Splitting images...", total=n_images):
        directory_name = "train" if i in train_indices else "val" if i in val_indices else "test"
        local_image_file_path = "/".join(image_file_path.split("/")[-3:])
        new_image_file_path = f"{SKY_FINDER_PATH}{directory_name}/{local_image_file_path}"
        os.makedirs(os.path.dirname(new_image_file_path), exist_ok=True)

        if os.path.exists(new_image_file_path):
            continue

        try:
            shutil.copy2(image_file_path, new_image_file_path)
        except Exception as e:
            print(f"❌ Failed to copy {image_file_path} to {new_image_file_path}: {e}.")
            continue

    print("✅ All images split into train, val, and test sets.")


def remove_data() -> None:
    """
    Remove the downloaded archives and extracted data.
    """
    print("▶️  Removing archives...")
    archive_paths = get_paths_recursive(
        folder_path=SKY_FINDER_ARCHIVES_PATH,
        match_pattern="*.zip",
        path_type="f",
        recursive=False,
    )
    for archive_path in archive_paths:
        os.remove(archive_path)
    os.rmdir(SKY_FINDER_ARCHIVES_PATH)

    print("▶️  Removing extracted data...")
    extracted_dir_paths = get_paths_recursive(
        folder_path=SKY_FINDER_EXTRACTED_PATH,
        match_pattern="*",
        path_type="d",
        recursive=False,
    )
    for extracted_dir_path in extracted_dir_paths:
        shutil.rmtree(extracted_dir_path, ignore_errors=True)
    os.rmdir(SKY_FINDER_EXTRACTED_PATH)

    print("▶️  Removing classified images...")
    shutil.rmtree(SKY_FINDER_IMAGES_PATH, ignore_errors=True)

    print("✅ All unused data removed.")


def parse_args() -> None:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Download sky finder archives.")

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
        help="Force re-download and extraction of archives",
    )

    parser.add_argument(
        "-r",
        "--remove-data",
        action="store_true",
        help="Remove archives and extracted data after processing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_workers = args.max_workers
    force = args.force
    remove_archives = args.remove_data

    download_archives(max_workers=max_workers, force=force)
    extract_archives(force=force)
    classify_extracted_images(force=force)
    split_classified_images(force=force)

    if remove_archives:
        remove_data()


if __name__ == "__main__":
    main()
