import os
import sys
import torch
import argparse
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.unet import UNet
from src.datasets.sky_finder_cover_dataset import SkyFinderCoverModule
from src.lightning_models.unet_lightning_model import UNetLightningModel
from src.config import (
    UNET_MANUAL_CHECKPOINT_PATH,
    UNET_ACTIVE_CHECKPOINT_PATH,
    DEVICE,
    SEED,
)


def get_continuous_iou(
    prediction: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-6
) -> float:
    """
    Calculate continuous IoU for regression outputs.
    """
    # Flatten tensors
    prediction = prediction.view(-1)
    ground_truth = ground_truth.view(-1)

    # Calculate intersection and union for continuous values
    intersection = (prediction * ground_truth).sum()
    union = prediction.sum() + ground_truth.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def get_continuous_dice(
    prediction: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-6
) -> float:
    """
    Calculate continuous Dice coefficient for regression outputs.
    """
    # Flatten tensors
    prediction = prediction.view(-1)
    ground_truth = ground_truth.view(-1)

    intersection = (prediction * ground_truth).sum()
    dice = (2.0 * intersection + smooth) / (
        prediction.sum() + ground_truth.sum() + smooth
    )

    return dice.item()


def get_coverage_error(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Calculate the absolute error in cloud coverage percentage.
    """
    pred_coverage = prediction.mean()
    gt_coverage = ground_truth.mean()
    error = torch.abs(pred_coverage - gt_coverage)
    return error.item()


def get_sky_class_error(pred_coverage: float, gt_coverage: float) -> float:
    """
    Calculate the absolute error in sky class prediction (0-1 range).
    """
    return abs(pred_coverage - gt_coverage)


def parse_args() -> None:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate UNet model.")

    parser.add_argument(
        "--active",
        "-a",
        action="store_true",
        help="Use active learning checkpoint for evaluation.",
    )
    parser.add_argument(
        "--with-pseudo-labelling",
        "-p",
        action="store_true",
        help="Use pseudo-labelling during evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active = args.active
    with_pseudo_labelling = args.with_pseudo_labelling

    checkpoint_path = UNET_ACTIVE_CHECKPOINT_PATH if active else UNET_MANUAL_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Checkpoint not found at {checkpoint_path}."
        )

    # Get models
    model = UNet(pretrained=True).to(DEVICE)

    lightning_model = UNetLightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()

    # Get dataloaders
    module = SkyFinderCoverModule(
        batch_size=1, n_workers=1, with_pseudo_labelling=with_pseudo_labelling, seed=SEED
    )
    module.setup(stage="fit")
    module.setup(stage="test")
    val_dataloader = module.val_dataloader()

    # Metrics storage
    ious = []
    dices = []
    coverage_errors = []
    sky_class_errors = []

    for batch in val_dataloader:
        image = batch[0][0]
        ground_truth = batch[1][0]
        gt_sky_class = (
            batch[2][0].item() if len(batch) > 2 else ground_truth.mean().item()
        )

        with torch.no_grad():
            # Regular model prediction
            seg_prediction, sky_class_pred = model(image.unsqueeze(0).to(DEVICE))
            seg_prediction = seg_prediction[0, 0, :, :].cpu()
            seg_prediction = torch.sigmoid(seg_prediction)
            sky_class_pred = torch.sigmoid(sky_class_pred[0].cpu()).item()

        # Normalize ground truth to [0, 1] if needed
        if ground_truth.max() > 1.0:
            ground_truth = ground_truth / ground_truth.max()

        # Calculate continuous metrics
        iou = get_continuous_iou(seg_prediction, ground_truth)
        dice = get_continuous_dice(seg_prediction, ground_truth)
        coverage_error = get_coverage_error(seg_prediction, ground_truth)
        sky_class_error = get_sky_class_error(sky_class_pred, gt_sky_class)

        ious.append(iou)
        dices.append(dice)
        coverage_errors.append(coverage_error)
        sky_class_errors.append(sky_class_error)

    # Calculate mean metrics
    mean_iou = sum(ious) / len(ious)
    mean_dice = sum(dices) / len(dices)
    mean_coverage_error = sum(coverage_errors) / len(coverage_errors)
    mean_sky_class_error = sum(sky_class_errors) / len(sky_class_errors)

    print("Results:")
    print(
        f"{'Model':<20} {'IoU':<8} {'Dice':<8} {'Coverage Err':<12} {'Sky Class Err':<12}"
    )
    print("-" * 75)
    print(
        f"{'Manual Labels':<20} {mean_iou:<8.4f} {mean_dice:<8.4f} {mean_coverage_error:<12.4f} {mean_sky_class_error:<12.4f}"
    )


if __name__ == "__main__":
    main()
