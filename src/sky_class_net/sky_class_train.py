import os
import sys
import time
import torch
import argparse
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.sky_class_net import SkyClassNet
from src.datasets.sky_finder_dataset import SkyFinderModule
from src.datasets.sky_finder_embeddings_dataset import SkyFinderEmbeddingsModule
from src.lightning_models.sky_class_lightning_model import SkyClassLightningModel
from src.utils.random import set_seed
from src.config import (
    MODELS_PATH,
    DEVICE,
    SEED,
)

N_EPOCHS = 100
BATCH_SIZE = 32
N_WORKERS = 1
EVALUATION_STEPS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0

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


def main() -> None:
    args = parse_args()
    contrastive_only = args.contrastive_only
    cover_only = args.cover_only
    if contrastive_only and cover_only:
        raise ValueError("❌ Cannot choose both --contrastive-only and --cover-only flags at the same time.")
    
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    # Get model
    input_dim = 17
    if contrastive_only:
        input_dim = 16
    elif cover_only:
        input_dim = 1

    model = SkyClassNet(
        input_dim=input_dim,
        output_dim=3,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)
    
    lightning_model = SkyClassLightningModel(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        name="sky_class_net",
        dataset="sky_finder_classification",
    )

    # Get trainer and train
    wandb_name = f"{time.strftime('%Y%m%d-%H%M%S')}_sky_class_net"
    config = {}
    wandb_logger = WandbLogger(
        project="lipid",
        name=wandb_name,
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{MODELS_PATH}/sky_class_net/{wandb_name}",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu",
        devices=-1,
        num_nodes=1,
        precision=32,
        strategy="auto",
        val_check_interval=EVALUATION_STEPS,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    embeddings_type = "all"
    if contrastive_only:
        embeddings_type = "contrastive"
    elif cover_only:
        embeddings_type = "cover"
    data_module = SkyFinderEmbeddingsModule(
        embeddings_type=embeddings_type,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        seed=SEED,
    )

    trainer.fit(
        model=lightning_model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()
