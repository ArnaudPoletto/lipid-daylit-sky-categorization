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

from src.models.unet import UNet
from src.utils.random import set_seed
from src.lightning_models.unet_lightning_model import UNetLightningModel
from src.datasets.sky_finder_cover_dataset import SkyFinderCoverModule
from src.config import (
    UNET_MANUAL_CHECKPOINT_PATH,
    MODELS_PATH,
    SEED,
    DEVICE,
)

N_EPOCHS = 100
BATCH_SIZE = 2
N_WORKERS = 8
EVALUATION_STEPS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BOTTLENECK_DROPOUT_RATE = 0.0
DECODER_DROPOUT_RATE = 0.0


def parse_args() -> None:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train UNet model.")

    parser.add_argument(
        "-a",
        "--active",
        action="store_true",
        help="Perform active learning training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active = args.active

    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    # Get model
    model = UNet(
        pretrained=True,
        bottleneck_dropout_rate=BOTTLENECK_DROPOUT_RATE,
        decoder_dropout_rate=DECODER_DROPOUT_RATE,
    ).to(DEVICE)
    if active:
        lightning_model = UNetLightningModel.load_from_checkpoint(
            UNET_MANUAL_CHECKPOINT_PATH,
            model=model,
            learning_rate=LEARNING_RATE * 0.1,
            weight_decay=WEIGHT_DECAY,
            name="unet",
            dataset="sky_finder_cover",
        )
        print(f"✅ Loaded model from {os.path.abspath(UNET_MANUAL_CHECKPOINT_PATH)}.")
    else:
        lightning_model = UNetLightningModel(
            model=model,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            name="unet",
            dataset="sky_finder_cover",
        )
        print("✅ Loaded pretrained model.")

    # Get trainer and train
    wandb_name = f"{time.strftime('%Y%m%d-%H%M%S')}_unet"
    config = {}
    wandb_logger = WandbLogger(
        project="lipid",
        name=wandb_name,
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{MODELS_PATH}/unet/{wandb_name}",
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

    data_module = SkyFinderCoverModule(
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        with_pseudo_labelling=active,
        seed=SEED,
    )

    trainer.fit(
        model=lightning_model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()