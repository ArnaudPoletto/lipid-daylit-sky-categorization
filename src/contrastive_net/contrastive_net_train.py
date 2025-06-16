import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import torch
import argparse
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.random import set_seed
from src.models.contrastive_net import ContrastiveNet
from src.lightning_models.contrastive_lightning_model import ContrastiveLightningModel
from src.datasets.contrastive_pairs_dataset import ContrastivePairsModule
from src.config import (
    PROJECTION_DIM,
    MODELS_PATH,
    SEED,
)


def create_model(
    projection_dim: int,
    pretrained: bool,
    normalize_embeddings: bool,
    learning_rate: float,
    weight_decay: float,
) -> ContrastiveLightningModel:
    """
    Create and initialize the contrastive model.

    Args:
        projection_dim (int): Dimension of the projection layer.
        pretrained (bool): Whether to use pretrained backbone.
        normalize_embeddings (bool): Whether to normalize embeddings.
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay for regularization.

    Returns:
        ContrastiveLightningModel: The initialized lightning model.
    """
    try:
        model = ContrastiveNet(
            projection_dim=projection_dim, 
            pretrained=pretrained,
            normalize_embeddings=normalize_embeddings,
        )
        lightning_model = ContrastiveLightningModel(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name="contrastive_net",
            dataset="sky_finder",
        )
        print("✅ Successfully created contrastive model.")
        return lightning_model
    except Exception as e:
        print(f"❌ Failed to create contrastive model: {e}")
        raise


def setup_wandb_logger(
    project_name: str,
    experiment_name: str,
    config: dict,
) -> WandbLogger:
    """
    Setup Weights & Biases logger.

    Args:
        project_name (str): Name of the W&B project.
        experiment_name (str): Name of the experiment.
        config (dict): Configuration dictionary to log.

    Returns:
        WandbLogger: The configured W&B logger.
    """
    try:
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            config=config,
        )
        print(f"✅ Successfully setup W&B logger for project '{project_name}'.")
        return wandb_logger
    except Exception as e:
        print(f"❌ Failed to setup W&B logger: {e}")
        raise


def setup_callbacks(
    checkpoint_dir: str,
    save_top_k: int,
    monitor_metric: str,
) -> list:
    """
    Setup training callbacks.

    Args:
        checkpoint_dir (str): Directory to save model checkpoints.
        save_top_k (int): Number of best checkpoints to save.
        monitor_metric (str): Metric to monitor for checkpointing.

    Returns:
        list: List of configured callbacks.
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=save_top_k,
            monitor=monitor_metric,
            mode="min",
        )
        callbacks = [checkpoint_callback]
        
        print(f"✅ Successfully setup callbacks. Checkpoints will be saved to {os.path.abspath(checkpoint_dir)}")
        return callbacks
    except Exception as e:
        print(f"❌ Failed to setup callbacks: {e}")
        raise


def setup_trainer(
    max_epochs: int,
    accelerator: str,
    devices: int,
    precision: int,
    val_check_interval: int,
    logger: WandbLogger,
    callbacks: list,
) -> pl.Trainer:
    """
    Setup PyTorch Lightning trainer.

    Args:
        max_epochs (int): Maximum number of training epochs.
        accelerator (str): Hardware accelerator to use.
        devices (int): Number of devices to use (-1 for all available).
        precision (int): Numerical precision for training.
        val_check_interval (int): How often to run validation.
        logger (WandbLogger): Logger for experiment tracking.
        callbacks (list): List of training callbacks.

    Returns:
        pl.Trainer: The configured trainer.
    """
    try:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            num_nodes=1,
            precision=precision,
            strategy="auto",
            val_check_interval=val_check_interval,
            logger=logger,
            callbacks=callbacks,
        )
        print("✅ Successfully setup PyTorch Lightning trainer.")
        return trainer
    except Exception as e:
        print(f"❌ Failed to setup trainer: {e}")
        raise


def setup_data_module(
    batch_size: int,
    n_workers: int,
    seed: int,
) -> ContrastivePairsModule:
    """
    Setup the data module for contrastive learning.

    Args:
        batch_size (int): Batch size for data loading.
        n_workers (int): Number of workers for data loading.
        seed (int): Random seed for reproducibility.

    Returns:
        ContrastivePairsModule: The configured data module.
    """
    try:
        data_module = ContrastivePairsModule(
            batch_size=batch_size,
            n_workers=n_workers,
            seed=seed,
        )
        print("✅ Successfully setup data module.")
        return data_module
    except Exception as e:
        print(f"❌ Failed to setup data module: {e}")
        raise


def train_model(
    trainer: pl.Trainer,
    lightning_model: ContrastiveLightningModel,
    data_module: ContrastivePairsModule,
) -> None:
    """
    Train the contrastive model.

    Args:
        trainer (pl.Trainer): The configured trainer.
        lightning_model (ContrastiveLightningModel): The model to train.
        data_module (ContrastivePairsModule): The data module.
    """
    try:
        print("▶️  Starting model training...")
        trainer.fit(
            model=lightning_model,
            datamodule=data_module,
        )
        print("✅ Model training completed successfully.")
    except Exception as e:
        print(f"❌ Failed during model training: {e}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train contrastive model for sky finder dataset.")

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs (default: 4)",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)",
    )

    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )

    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=500,
        help="Number of steps between validation runs (default: 500)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimization (default: 1e-4)",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 1e-4)",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="lipid",
        help="W&B project name (default: lipid)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name (default: auto-generated timestamp)",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu"],
        help="Hardware accelerator to use (default: gpu)",
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=-1,
        help="Number of devices to use, -1 for all available (default: -1)",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Training precision (default: 32)",
    )

    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Number of best checkpoints to save (default: 3)",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly initialized backbone instead of pretrained",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding normalization",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to train the contrastive model.
    """
    args = parse_args()

    # Generate experiment name if not provided
    if args.experiment_name is None:
        experiment_name = f"{time.strftime('%Y%m%d-%H%M%S')}_contrastive_net"
    else:
        experiment_name = args.experiment_name

    # Setup environment
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    print("▶️  Starting contrastive model training...")
    print(f"📋 Configuration:")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Workers: {args.n_workers}")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • Weight decay: {args.weight_decay}")
    print(f"   • Evaluation steps: {args.evaluation_steps}")
    print(f"   • Accelerator: {args.accelerator}")
    print(f"   • Devices: {args.devices}")
    print(f"   • Precision: {args.precision}")
    print(f"   • Pretrained backbone: {not args.no_pretrained}")
    print(f"   • Normalize embeddings: {not args.no_normalize}")
    print(f"   • Projection dimension: {PROJECTION_DIM}")
    print(f"   • Save top K checkpoints: {args.save_top_k}")
    print(f"   • W&B project: {args.project_name}")
    print(f"   • Experiment name: {experiment_name}")
    print(f"   • Models path: {os.path.abspath(MODELS_PATH)}")
    print(f"   • Random seed: {SEED}")

    try:
        # Create model
        print("▶️  Creating contrastive model...")
        lightning_model = create_model(
            projection_dim=PROJECTION_DIM,
            pretrained=not args.no_pretrained,
            normalize_embeddings=not args.no_normalize,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Setup experiment tracking
        print("▶️  Setting up experiment tracking...")
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "projection_dim": PROJECTION_DIM,
            "pretrained": not args.no_pretrained,
            "normalize_embeddings": not args.no_normalize,
            "evaluation_steps": args.evaluation_steps,
            "seed": SEED,
        }
        wandb_logger = setup_wandb_logger(
            project_name=args.project_name,
            experiment_name=experiment_name,
            config=config,
        )

        # Setup callbacks
        print("▶️  Setting up training callbacks...")
        checkpoint_dir = f"{MODELS_PATH}/contrastive_net/{experiment_name}"
        callbacks = setup_callbacks(
            checkpoint_dir=checkpoint_dir,
            save_top_k=args.save_top_k,
            monitor_metric="val_loss",
        )

        # Setup trainer
        print("▶️  Setting up trainer...")
        trainer = setup_trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            val_check_interval=args.evaluation_steps,
            logger=wandb_logger,
            callbacks=callbacks,
        )

        # Setup data
        print("▶️  Setting up data module...")
        data_module = setup_data_module(
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            seed=SEED,
        )

        # Train model
        train_model(
            trainer=trainer,
            lightning_model=lightning_model,
            data_module=data_module,
        )

        print("🎉 Contrastive model training completed successfully!")
        print(f"📁 Checkpoints saved to: {os.path.abspath(checkpoint_dir)}")

    except Exception as e:
        print(f"💥 Fatal error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()