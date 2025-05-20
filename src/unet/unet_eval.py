import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.unet import UNet
from src.datasets.sky_finder_cover_dataset import SkyFinderCoverModule
from src.lightning_models.unet_lightning_model import UNetLightningModel
from src.config import (
    UNET_ACTIVE_CHECKPOINT_PATH,
    UNET_CHECKPOINT_PATH,
    DEVICE,
    SEED,
)

def get_iou(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Calculate the Intersection over Union (IoU) between the prediction and ground truth.
    """
    intersection = torch.logical_and(prediction, ground_truth).float().sum()
    union = torch.logical_or(prediction, ground_truth).float().sum()

    if union == 0:
        return 1.0
    
    iou = intersection / union
    return iou.item()

def get_dice(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Calculate the Dice coefficient between the prediction and ground truth.
    """
    intersection = torch.logical_and(prediction, ground_truth).float().sum()

    denominator = prediction.float().sum() + ground_truth.float().sum()
    if denominator == 0:
        return 1.0
    
    dice = (2 * intersection) / denominator
    return dice.item()

def get_accuracy(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Calculate the accuracy between the prediction and ground truth.
    """
    correct = (prediction == ground_truth).float().sum()
    total = prediction.numel()
    accuracy = correct / total
    return accuracy.item()

def get_f1(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Calculate the F1 score between the prediction and ground truth.
    """
    intersection = torch.logical_and(prediction, ground_truth).float().sum()

    pred_sum = prediction.float().sum()
    gt_sum = ground_truth.float().sum()
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    
    precision = intersection / pred_sum
    recall = intersection / gt_sum
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1.item()

def main() -> None:
    if not os.path.exists(UNET_CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ Checkpoint not found at {UNET_CHECKPOINT_PATH}.")
    if not os.path.exists(UNET_ACTIVE_CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ Checkpoint not found at {UNET_ACTIVE_CHECKPOINT_PATH}.")
    
    # Get model
    model = UNet(pretrained=True).to(DEVICE)
    active_model = UNet(pretrained=True).to(DEVICE)
    
    lightning_model = UNetLightningModel.load_from_checkpoint(
        UNET_CHECKPOINT_PATH,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    lightning_active_model = UNetLightningModel.load_from_checkpoint(
        UNET_ACTIVE_CHECKPOINT_PATH,
        model=active_model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()
    active_model = lightning_active_model.model.to(DEVICE)
    active_model.eval()

    # Get dataloaders
    module = SkyFinderCoverModule(
        batch_size=1,
        n_workers=1,
        with_pseudo_labelling=False,
        seed=SEED
    )
    module.setup(stage="fit")
    module.setup(stage="test")
    val_dataloader = module.val_dataloader()

    ious = []
    dices = []
    accuracies = []
    f1s = []
    active_ious = []
    active_dices = []
    active_accuracies = []
    active_f1s = []
    for batch in val_dataloader:
        image = batch[0][0]
        ground_truth = batch[1][0]

        with torch.no_grad():
            prediction = model(image.unsqueeze(0).to(DEVICE))
            prediction = prediction[0, 0, :, :].cpu()
            prediction = torch.sigmoid(prediction)

            active_prediction = active_model(image.unsqueeze(0).to(DEVICE))
            active_prediction = active_prediction[0, 0, :, :].cpu()
            active_prediction = torch.sigmoid(active_prediction)

        # Binarize predictions
        prediction = (prediction > 0.5).float()
        active_prediction = (active_prediction > 0.5).float()
        ground_truth = (ground_truth > 0.5).float()

        # Calculate metrics
        iou = get_iou(prediction, ground_truth)
        dice = get_dice(prediction, ground_truth)
        accuracy = get_accuracy(prediction, ground_truth)
        f1 = get_f1(prediction, ground_truth)
        ious.append(iou)
        dices.append(dice)
        accuracies.append(accuracy)
        f1s.append(f1)

        active_iou = get_iou(active_prediction, ground_truth)
        active_dice = get_dice(active_prediction, ground_truth)
        active_accuracy = get_accuracy(active_prediction, ground_truth)
        active_f1 = get_f1(active_prediction, ground_truth)
        active_ious.append(active_iou)
        active_dices.append(active_dice)
        active_accuracies.append(active_accuracy)
        active_f1s.append(active_f1)

    # Calculate mean metrics
    mean_iou = sum(ious) / len(ious)
    mean_dice = sum(dices) / len(dices)
    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_f1 = sum(f1s) / len(f1s)
    mean_active_iou = sum(active_ious) / len(active_ious)
    mean_active_dice = sum(active_dices) / len(active_dices)
    mean_active_accuracy = sum(active_accuracies) / len(active_accuracies)
    mean_active_f1 = sum(active_f1s) / len(active_f1s)

    print("Results:")
    print(f"\tMean IoU: {mean_iou:.4f}")
    print(f"\tMean Dice: {mean_dice:.4f}")
    print(f"\tMean Accuracy: {mean_accuracy:.4f}")
    print(f"\tMean F1: {mean_f1:.4f}")
    print(f"\t---")
    print(f"\tMean Active IoU: {mean_active_iou:.4f}")
    print(f"\tMean Active Dice: {mean_active_dice:.4f}")
    print(f"\tMean Active Accuracy: {mean_active_accuracy:.4f}")
    print(f"\tMean Active F1: {mean_active_f1:.4f}")

if __name__ == "__main__":
    main()