import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.unet import UNet
from src.models.contrastive_net import ContrastiveNet
from src.datasets.sky_finder_dataset import SkyFinderModule
from src.lightning_models.unet_lightning_model import UNetLightningModel
from src.lightning_models.contrastive_lightning_model import ContrastiveLightningModel
from src.config import (
    SKY_FINDER_DESCRIPTORS_PATH,
    CONTRASTIVE_CHECKPOINT_PATH,
    UNET_ACTIVE_CHECKPOINT_PATH,
    PROJECTION_DIM,
    DEVICE,
    SEED,
)


def load_contrastive_model() -> ContrastiveNet:
    """
    Load the pretrained contrastive model.

    Returns:
        ContrastiveNet: The loaded contrastive model.
    """
    try:
        contrastive_model = ContrastiveNet(
            projection_dim=PROJECTION_DIM, 
            pretrained=True, 
            normalize_embeddings=False,
        )
        lightning_contrastive_model = ContrastiveLightningModel.load_from_checkpoint(
            CONTRASTIVE_CHECKPOINT_PATH,
            model=contrastive_model,
            learning_rate=0,
            weight_decay=0,
            name="contrastive_net",
            dataset="sky_finder",
        )
        contrastive_model = lightning_contrastive_model.model.to(DEVICE)
        contrastive_model.eval()
        print("✅ Successfully loaded contrastive model.")
        return contrastive_model
    except Exception as e:
        print(f"❌ Failed to load contrastive model: {e}")
        raise


def load_cover_model() -> UNet:
    """
    Load the pretrained cloud cover UNet model.

    Returns:
        UNet: The loaded cloud cover model.
    """
    try:
        cover_model = UNet(pretrained=True).to(DEVICE)
        lightning_cover_model = UNetLightningModel.load_from_checkpoint(
            UNET_ACTIVE_CHECKPOINT_PATH,
            model=cover_model,
            learning_rate=0,
            weight_decay=0,
            name="unet",
            dataset="sky_finder_cover",
        )
        cover_model = lightning_cover_model.model.to(DEVICE)
        cover_model.eval()
        print("✅ Successfully loaded cloud cover model.")
        return cover_model
    except Exception as e:
        print(f"❌ Failed to load cloud cover model: {e}")
        raise


def setup_data_module(
    batch_size: int,
    n_workers: int,
) -> SkyFinderModule:
    """
    Setup the data module with train, validation, and test dataloaders.

    Args:
        batch_size (int): Batch size for data loading.
        n_workers (int): Number of workers for data loading.

    Returns:
        SkyFinderModule: The configured data module.
    """
    try:
        data_module = SkyFinderModule(
            batch_size=batch_size,
            n_workers=n_workers,
            seed=SEED,
        )
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        print("✅ Successfully setup data module.")
        return data_module
    except Exception as e:
        print(f"❌ Failed to setup data module: {e}")
        raise


def generate_descriptors_for_split(
    dataloader: DataLoader,
    contrastive_model: ContrastiveNet,
    cover_model: UNet,
    split_name: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]]:
    """
    Generate descriptors for a specific data split.
    
    Args:
        dataloader (DataLoader): DataLoader for the dataset split.
        contrastive_model (ContrastiveNet): Pretrained contrastive model for image embeddings.
        cover_model (UNet): Pretrained UNet model for cloud cover prediction.
        split_name (str): Name of the data split (train/val/test).
    
    Returns:
        Dict: A nested dictionary containing descriptors for each image in the split.
    """
    descriptors = {}
    
    for batch in tqdm(dataloader, desc=f"⏳ Generating {split_name} descriptors..."):
        try:
            dataset_split = batch[0][0]
            sky_class = batch[1][0]
            camera_id = batch[2][0]
            image_name = batch[3][0]
            mask = batch[4][0]
            image = batch[5][0]
            image = image.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # Get sky image descriptor
                sky_image_descriptor = contrastive_model(image).cpu().numpy()[0].tolist()

                # Get cloud cover prediction
                cloud_coverage, _ = cover_model(image)
                cloud_coverage = cloud_coverage[0, 0, :, :].detach().cpu().numpy()
                mask = mask.cpu().numpy()
                mask = torch.nn.functional.interpolate(
                    torch.tensor(mask).unsqueeze(0).unsqueeze(0).float(),
                    size=cloud_coverage.shape,
                    mode='nearest',
                ).squeeze().numpy() > 0.5
                cloud_coverage = cloud_coverage[mask]
                mean_cloud_coverage = float(cloud_coverage.mean())

                # Build nested dictionary structure
                if dataset_split not in descriptors:
                    descriptors[dataset_split] = {}
                if sky_class not in descriptors[dataset_split]:
                    descriptors[dataset_split][sky_class] = {}
                if camera_id not in descriptors[dataset_split][sky_class]:
                    descriptors[dataset_split][sky_class][camera_id] = {}
                if image_name not in descriptors[dataset_split][sky_class][camera_id]:
                    descriptors[dataset_split][sky_class][camera_id][image_name] = {}
                
                descriptors[dataset_split][sky_class][camera_id][image_name].update({
                    "sky_image_descriptor": sky_image_descriptor,
                    "cloud_coverage": mean_cloud_coverage,
                })

        except Exception as e:
            print(f"❌ Failed to process batch in {split_name}: {e}")
            continue

    return descriptors


def generate_all_descriptors(
    data_module: SkyFinderModule,
    contrastive_model: ContrastiveNet,
    cover_model: UNet,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]]:
    """
    Generate descriptors for all data splits (train, validation, test).

    Args:
        data_module (SkyFinderModule): The data module containing all dataloaders.
        contrastive_model (ContrastiveNet): Pretrained contrastive model for image embeddings.
        cover_model (UNet): Pretrained UNet model for cloud cover prediction.

    Returns:
        Dict: A nested dictionary containing all descriptors.
    """
    print("▶️  Generating descriptors for all data splits...")

    # Generate descriptors for each split
    train_descriptors = generate_descriptors_for_split(
        dataloader=data_module.train_dataloader(),
        contrastive_model=contrastive_model,
        cover_model=cover_model,
        split_name="train",
    )

    val_descriptors = generate_descriptors_for_split(
        dataloader=data_module.val_dataloader(),
        contrastive_model=contrastive_model,
        cover_model=cover_model,
        split_name="validation",
    )

    test_descriptors = generate_descriptors_for_split(
        dataloader=data_module.test_dataloader(),
        contrastive_model=contrastive_model,
        cover_model=cover_model,
        split_name="test",
    )

    # Merge all descriptors
    all_descriptors = {}
    for descriptor_dict in [train_descriptors, val_descriptors, test_descriptors]:
        for split_key, split_data in descriptor_dict.items():
            if split_key not in all_descriptors:
                all_descriptors[split_key] = {}
            all_descriptors[split_key].update(split_data)

    print("✅ All descriptors generated successfully.")
    return all_descriptors


def save_descriptors(
    descriptors: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]],
    force: bool,
) -> None:
    """
    Save the generated descriptors to a JSON file.

    Args:
        descriptors (Dict): The descriptors dictionary to save.
        force (bool): Whether to force overwrite existing descriptor file.
    """
    print("▶️  Saving descriptors...")

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(SKY_FINDER_DESCRIPTORS_PATH), exist_ok=True)
        
        # Save descriptors to JSON file
        with open(SKY_FINDER_DESCRIPTORS_PATH, "w") as f:
            json.dump(descriptors, f, indent=4)
        
        print(f"✅ Descriptors saved successfully to {os.path.abspath(SKY_FINDER_DESCRIPTORS_PATH)}")
        
        # Print summary statistics
        total_images = 0
        for split_data in descriptors.values():
            for class_data in split_data.values():
                for camera_data in class_data.values():
                    total_images += len(camera_data)
        print(f"📊 Generated descriptors for {total_images} images across {len(descriptors)} splits.")
        
    except Exception as e:
        print(f"❌ Failed to save descriptors: {e}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate sky finder descriptors using pretrained models.")

    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=1,
        help="Number of workers for data loading (default: 1)",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite existing descriptor file",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to generate sky finder descriptors.
    """
    args = parse_args()
    
    print("▶️  Starting sky finder descriptor generation...")
    print(f"📋 Configuration:")
    print(f"   • Workers: {args.n_workers}")
    print(f"   • Force overwrite: {args.force}")
    print(f"   • Contrastive checkpoint: {os.path.abspath(CONTRASTIVE_CHECKPOINT_PATH)}")
    print(f"   • UNet checkpoint: {os.path.abspath(UNET_ACTIVE_CHECKPOINT_PATH)}")
    print(f"   • Output path: {os.path.abspath(SKY_FINDER_DESCRIPTORS_PATH)}")
    print(f"   • Device: {DEVICE}")

    # Check if output file already exists and handle force flag early
    if os.path.exists(SKY_FINDER_DESCRIPTORS_PATH) and not args.force:
        print(f"⚠️  Descriptors file already exists at {os.path.abspath(SKY_FINDER_DESCRIPTORS_PATH)}")
        print("Use --force to overwrite existing file.")
        sys.exit(0)

    try:
        # Load models
        print("▶️  Loading pretrained models...")
        contrastive_model = load_contrastive_model()
        cover_model = load_cover_model()

        # Setup data
        print("▶️  Setting up data module...")
        data_module = setup_data_module(
            batch_size=1,
            n_workers=args.n_workers,
        )

        # Generate descriptors
        descriptors = generate_all_descriptors(
            data_module=data_module,
            contrastive_model=contrastive_model,
            cover_model=cover_model,
        )

        # Save descriptors
        save_descriptors(
            descriptors=descriptors,
            force=args.force,
        )

        print("🎉 Sky finder descriptor generation completed successfully!")

    except Exception as e:
        print(f"💥 Fatal error during descriptor generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()