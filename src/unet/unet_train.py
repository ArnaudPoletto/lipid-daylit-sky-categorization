import os
import sys
import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.unet import UNet
from src.utils.random import set_seed
from src.lightning_models.unet_lightning_model import UNetLightningModel
from src.datasets.sky_cover_dataset import SkyCoverModule
from src.config import (
    MODELS_PATH,
    SEED,
)

N_EPOCHS = 20
BATCH_SIZE = 2
N_WORKERS = 8
EVALUATION_STEPS = 250
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def main() -> None:
    set_seed(SEED)

    torch.set_float32_matmul_precision("high")

    # Get model
    model = UNet(pretrained=True)
    lightning_model = UNetLightningModel(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        name="unet",
        dataset="sky_cover",
    )

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

    data_module = SkyCoverModule(
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
