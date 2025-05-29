import os
import sys
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.sky_class_net import SkyClassNet
from src.datasets.sky_finder_embeddings_dataset import SkyFinderEmbeddingsModule
from src.lightning_models.sky_class_lightning_model import SkyClassLightningModel
from src.config import (
    CONTRASTIVE_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH,
    COVER_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH,
    ALL_SKY_CLASS_NET_CHECKPOINT_PATH,
    DEVICE,
    SEED,
)   

def evaluate(model, dataloader, stage: str):
    """
    Evaluate the model on the given dataloader and print accuracy and F1 score.
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    confusion_matrix = torch.zeros(3, 3, dtype=torch.int64)
    for batch in dataloader:
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y = torch.argmax(y, dim=1)

        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)

        TP += torch.sum((y_pred == 1) & (y == 1)).item()
        TN += torch.sum((y_pred == 0) & (y == 0)).item()
        FP += torch.sum((y_pred == 1) & (y == 0)).item()
        FN += torch.sum((y_pred == 0) & (y == 1)).item()
        for i in range(3):
            for j in range(3):
                confusion_matrix[i, j] += torch.sum((y_pred == i) & (y == j)).item()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Stage: {stage} | Accuracy: {accuracy:.4f} | F1 Score: {f1_score:.4f} | TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print("Confusion Matrix:")
    print(confusion_matrix)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train Sky Class Net.")

    parser.add_argument(
        "--contrastive-only",
        action="store_true",
        help="Use contrastive embeddings only for training.",
    )
    parser.add_argument(
        "--cover-only",
        action="store_true",
        help="Use cover embeddings only for training.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    contrastive_only = args.contrastive_only
    cover_only = args.cover_only
    if contrastive_only and cover_only:
        raise ValueError("❌ Cannot choose both --contrastive-only and --cover-only flags at the same time.")

    # Get model
    input_dim = 17
    if contrastive_only:
        input_dim = 16
    elif cover_only:
        input_dim = 1

    input_dim = 17
    if contrastive_only:
        input_dim = 16
    elif cover_only:
        input_dim = 1

    model = SkyClassNet(
        input_dim=input_dim,
        output_dim=3,
        dropout_rate=0.0,
    ).to(DEVICE)

    model_path = ALL_SKY_CLASS_NET_CHECKPOINT_PATH
    if contrastive_only:
        model_path = CONTRASTIVE_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH
    elif cover_only:
        model_path = COVER_ONLY_SKY_CLASS_NET_CHECKPOINT_PATH

    lightning_model = SkyClassLightningModel.load_from_checkpoint(
        model_path,
        model=model,
        learning_rate=0,
        weight_decay=0,
        name="unet",
        dataset="sky_finder_cover",
    )
    model = lightning_model.model.to(DEVICE)
    model.eval()

    # Get dataloaders
    embeddings_type = "all"
    if contrastive_only:
        embeddings_type = "contrastive"
    elif cover_only:
        embeddings_type = "cover"
    module = SkyFinderEmbeddingsModule(
        embeddings_type=embeddings_type,
        batch_size=1,
        n_workers=1,
        seed=SEED,
    )
    module.setup(stage="fit")
    module.setup(stage="test")
    train_dataloader = module.train_dataloader()
    val_dataloader = module.val_dataloader()
    test_dataloader = module.test_dataloader()

    evaluate(model, train_dataloader, "train")
    evaluate(model, val_dataloader, "val")
    evaluate(model, test_dataloader, "test")

if __name__ == "__main__":
    main()