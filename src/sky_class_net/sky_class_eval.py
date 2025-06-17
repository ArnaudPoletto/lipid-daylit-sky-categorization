import os
import sys
import torch
import argparse
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.sky_class_net import SkyClassNet
from src.datasets.sky_finder_embeddings_dataset import SkyFinderEmbeddingsModule
from src.lightning_models.sky_class_lightning_model import SkyClassLightningModel
from src.config import (
    SKY_CLASS_NET_CHECKPOINT_PATH,
    PROJECTION_DIM,
    DEVICE,
    SEED,
)


def load_model(
    model_path: str,
) -> SkyClassNet:
    """
    Load the trained sky classification model from checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.

    Returns:
        SkyClassNet: The loaded model.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        model = SkyClassNet(
            input_dim=PROJECTION_DIM,
            output_dim=3,
            dropout_rate=0.0,
        ).to(DEVICE)

        lightning_model = SkyClassLightningModel.load_from_checkpoint(
            model_path,
            model=model,
            learning_rate=0,
            weight_decay=0,
            name="sky_class_net",
            dataset="sky_finder_classification",
        )
        model = lightning_model.model.to(DEVICE)
        model.eval()
        print(f"✅ Successfully loaded model from {os.path.abspath(model_path)}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def setup_data_module(
    n_workers: int,
) -> SkyFinderEmbeddingsModule:
    """
    Setup the data module for evaluation.

    Args:
        n_workers (int): Number of workers for data loading.

    Returns:
        SkyFinderEmbeddingsModule: The configured data module.
    """
    try:
        module = SkyFinderEmbeddingsModule(
            batch_size=1,
            n_workers=n_workers,
            seed=SEED,
        )
        module.setup(stage="fit")
        module.setup(stage="test")
        return module
    except Exception as e:
        print(f"❌ Failed to setup data module: {e}")
        raise


def calculate_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int = 3,
) -> Tuple[float, float, torch.Tensor]:
    """
    Calculate evaluation metrics from predictions.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.

    Returns:
        Tuple[float, float, torch.Tensor]: (accuracy, f1_score, confusion_matrix)
    """
    # Calculate confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[j, i] = torch.sum((y_pred == i) & (y_true == j)).item()

    # Calculate overall accuracy
    correct = torch.sum(y_pred == y_true).item()
    total = y_true.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # Calculate macro F1 score
    f1_scores = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i].item()
        fp = torch.sum(confusion_matrix[i, :]).item() - tp
        fn = torch.sum(confusion_matrix[:, i]).item() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    return accuracy, macro_f1, confusion_matrix


def evaluate_model(
    model: SkyClassNet,
    dataloader: DataLoader,
    stage: str,
) -> Dict[str, Any]:
    """
    Evaluate the model on the given dataloader.

    Args:
        model (SkyClassNet): The model to evaluate.
        dataloader (DataLoader): The data loader for evaluation.
        stage (str): Evaluation stage name (train/val/test).

    Returns:
        Dict[str, Any]: Evaluation results.
    """
    try:
        print(f"▶️  Evaluating on {stage} set...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                
                # Convert one-hot to class indices if needed
                if y.dim() > 1 and y.size(1) > 1:
                    y = torch.argmax(y, dim=1)
                
                # Get predictions
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)
                
                all_predictions.append(y_pred.cpu())
                all_targets.append(y.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Calculate metrics
        accuracy, f1_score, confusion_matrix = calculate_metrics(all_targets, all_predictions)
        
        # Print results
        print(f"✅ {stage.capitalize()} Results:")
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • F1 Score: {f1_score:.4f}")
        print(f"   • Confusion Matrix:")
        for i in range(confusion_matrix.size(0)):
            print(f"     {confusion_matrix[i].tolist()}")
        
        return {
            "stage": stage,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "confusion_matrix": confusion_matrix.tolist(),
            "total_samples": len(all_targets),
        }
        
    except Exception as e:
        print(f"❌ Failed to evaluate on {stage} set: {e}")
        raise


def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary table of all evaluation results.

    Args:
        results (Dict[str, Dict[str, Any]]): Results from all evaluation stages.
    """
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Stage':<10} {'Samples':<10} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*60)
    
    for stage_name, stage_results in results.items():
        print(f"{stage_name.capitalize():<10} "
              f"{stage_results['total_samples']:<10} "
              f"{stage_results['accuracy']:<12.4f} "
              f"{stage_results['f1_score']:<12.4f}")
    
    print("="*60)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained sky classification model.")

    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=1,
        help="Number of data loading workers (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to evaluate the sky classification model.
    """
    args = parse_args()

    # Determine configuration

    print("▶️  Starting sky classification model evaluation...")
    print(f"📋 Configuration:")
    print(f"   • Model checkpoint: {os.path.abspath(SKY_CLASS_NET_CHECKPOINT_PATH)}")
    print(f"   • Workers: {args.n_workers}")
    print(f"   • Device: {DEVICE}")
    print(f"   • Random seed: {SEED}")

    try:
        # Load model
        print("▶️  Loading trained model...")
        model = load_model(model_path=SKY_CLASS_NET_CHECKPOINT_PATH)

        # Setup data module
        print("▶️  Setting up data module...")
        data_module = setup_data_module(
            n_workers=args.n_workers,
        )

        # Get dataloaders
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()

        # Evaluate on all splits
        results = {}
        results["train"] = evaluate_model(model, train_dataloader, "train")
        results["val"] = evaluate_model(model, val_dataloader, "val")
        results["test"] = evaluate_model(model, test_dataloader, "test")

        # Print summary
        print_summary_table(results)

        print("🎉 Sky classification model evaluation completed successfully!")

    except Exception as e:
        print(f"💥 Fatal error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()