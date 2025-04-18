import os
import sys
import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.random import set_seed
from src.models.contrastive_net import ContrastiveNet
from src.utils.lightning_model import LightningModel
from src.datasets.contrastive_pairs_dataset import ContrastivePairsModule
from src.config import (
    EVALUATION_STEPS,
    PROJECTION_DIM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODELS_PATH,
    BATCH_SIZE,
    N_WORKERS,
    N_EPOCHS,
    SEED,
)


def main() -> None:
    set_seed(SEED)

    torch.set_float32_matmul_precision("high")

    # Get model
    model = ContrastiveNet(projection_dim=PROJECTION_DIM, pretrained=True)
    lightning_model = LightningModel(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        name="contrastive_net",
        dataset="sky_finder",
    )

    # Get trainer and train
    wandb_name = f"{time.strftime('%Y%m%d-%H%M%S')}_contrastive_net"
    config = {}
    wandb_logger = WandbLogger(
        project="lipid",
        name=wandb_name,
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{MODELS_PATH}/contrastive_net/{wandb_name}",
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

    data_module = ContrastivePairsModule(
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
