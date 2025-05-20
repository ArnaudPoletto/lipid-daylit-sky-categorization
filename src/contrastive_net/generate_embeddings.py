import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.sky_finder import (
    get_sky_finder_masks,
    get_sky_finder_bounding_boxes,
    get_sky_finder_paths_dict,
)
from src.models.contrastive_net import ContrastiveNet
from src.lightning_models.contrastive_lightning_model import ContrastiveLightningModel
from src.utils.file import get_paths_recursive
from src.config import (
    EMBEDDINGS_FILE_PATH,
    CONTRASTIVE_CHECKPOINT_PATH,
    PROJECTION_DIM,
    SKY_FINDER_WIDTH,
    SKY_FINDER_HEIGHT,
    DEVICE,
)


def get_model(checkpoint_path: str) -> ContrastiveNet:
    """
    Get the ContrastiveNet model from the checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        ContrastiveNet: ContrastiveNet model.
    """
    model = ContrastiveNet(projection_dim=PROJECTION_DIM, pretrained=True)
    lightning_model = ContrastiveLightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="contrastive_net",
        dataset="sky_finder",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()

    return model


def get_image(
    image_file_path: str,
    mask: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    mean: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32),
    std: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32),
) -> np.ndarray:
    """
    Get image from file path.

    Args:
        image_file_path (str): Path to the image file.
        mask (np.ndarray): Mask for the image.
        bounding_box (Tuple[int, int, int, int]): Bounding box for cropping the image.
        mean (np.ndarray, optional): Mean for normalization. Defaults to [0.485, 0.456, 0.406].
        std (np.ndarray, optional): Standard deviation for normalization. Defaults to [0.229, 0.224, 0.225].

    Returns:
        np.ndarray: Image as a numpy array.
    """
    # Read image
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"❌ Failed to read image: {image_file_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop image and mask based on bounding box
    x_min, y_min, x_max, y_max = bounding_box
    image = image[y_min:y_max, x_min:x_max]
    mask = mask[y_min:y_max, x_min:x_max]

    # Apply inpainting to fill the ground
    inpaint_mask = (~mask).astype(np.uint8) * 255
    image = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)

    # Resize and normalize image
    image = cv2.resize(image, (SKY_FINDER_WIDTH, SKY_FINDER_HEIGHT))
    image = image / 255.0
    image = (image - mean) / std

    return image


def get_embedding(
    image: np.ndarray,
    model: ContrastiveNet,
) -> np.ndarray:
    """
    Get embedding for the given image using the model.

    Args:
        image (np.ndarray): Image as a numpy array.
        model (ContrastiveNet): ContrastiveNet model.

    Returns:
        np.ndarray: Embedding as a numpy array.
    """
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.ascontiguousarray(image)  # Ensure contiguous memory layout
    image = torch.tensor(image, dtype=torch.float32).to(DEVICE)

    # Get embedding
    with torch.no_grad():
        embedding = model(image).cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = np.squeeze(embedding).tolist()

    return embedding


def parse_args() -> None:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate embeddings for images.")

    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=CONTRASTIVE_CHECKPOINT_PATH,
        help="Path to the model checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint_path

    # Get model
    model = get_model(checkpoint_path=checkpoint_path)

    # Generate embeddings
    paths_dict = get_sky_finder_paths_dict(split="test")
    masks = get_sky_finder_masks(paths_dict)
    bounding_boxes = get_sky_finder_bounding_boxes(paths_dict)
    embeddings = {}
    n_images = sum(
        len(image_paths) for camera_id in paths_dict.keys() for image_paths in paths_dict[camera_id].values()
    )
    bar = tqdm(total=n_images, desc="⌛ Generating embeddings...", unit="file")
    for sky_type in paths_dict.keys():
        for camera_id in paths_dict[sky_type].keys():
            image_file_paths = paths_dict[sky_type][camera_id]
            for image_file_path in image_file_paths:
                if camera_id not in masks:
                    print(f"❌ Camera ID {camera_id} not found in masks.")
                    bar.update(1)
                    continue
                mask = masks[camera_id]

                if camera_id not in bounding_boxes:
                    print(f"❌ Camera ID {camera_id} not found in bounding boxes.")
                    bar.update(1)
                    continue
                bounding_box = bounding_boxes[camera_id]

                image = get_image(
                    image_file_path=image_file_path, mask=mask, bounding_box=bounding_box
                )
                embedding = get_embedding(image=image, model=model)
                embeddings[image_file_path] = {"sky_type": sky_type, "embedding": embedding}

                bar.update(1)

    # Save embeddings
    with open(EMBEDDINGS_FILE_PATH, "w") as f:
        json.dump(embeddings, f, indent=4)
    print(f"✅ Saved embeddings to {os.path.abspath(EMBEDDINGS_FILE_PATH)}.")


if __name__ == "__main__":
    main()
