import os
import sys
import json
import torch
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

def generate_descriptors(
        dataloader: DataLoader,
        contrastive_model: ContrastiveNet,
        cover_model: UNet, 
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]]:
    descriptors = {}
    for batch in tqdm(dataloader, desc="⏳ Generating descriptors..."):
        dataset_split = batch[0][0]
        sky_class = batch[1][0]
        camera_id = batch[2][0]
        image_name = batch[3][0]
        mask = batch[4][0]
        image = batch[5][0]
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            contrastive_embeddings = contrastive_model(image).cpu().numpy()[0].tolist()
            cover_prediction = cover_model(image)[0, 0, :, :].detach().cpu().numpy()
            mean_cover_prediction = float(cover_prediction.mean())

            if dataset_split not in descriptors:
                descriptors[dataset_split] = {}
            if sky_class not in descriptors[dataset_split]:
                descriptors[dataset_split][sky_class] = {}
            if camera_id not in descriptors[dataset_split][sky_class]:
                descriptors[dataset_split][sky_class][camera_id] = {}
            if image_name not in descriptors[dataset_split][sky_class][camera_id]:
                descriptors[dataset_split][sky_class][camera_id][image_name] = {}
            descriptors[dataset_split][sky_class][camera_id][image_name].update({
                "contrastive_embeddings": contrastive_embeddings,
                "cover_prediction": mean_cover_prediction,
            })

    return descriptors

def main():
    # Get contrastive model
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

    # Get coverage estimator model
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

    # Get test data
    data_module = SkyFinderModule(
        batch_size=1,
        n_workers=1,
        seed=SEED,
    )
    data_module.setup(stage="fit")
    data_module.setup(stage="test")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Generate descriptors
    train_descriptors = generate_descriptors(
        dataloader=train_dataloader,
        contrastive_model=contrastive_model,
        cover_model=cover_model,
    )
    val_descriptors = generate_descriptors(
        dataloader=val_dataloader,
        contrastive_model=contrastive_model,
        cover_model=cover_model,
    )
    test_descriptors = generate_descriptors(
        dataloader=test_dataloader,
        contrastive_model=contrastive_model,
        cover_model=cover_model,
    )
    descriptors = train_descriptors | val_descriptors | test_descriptors
    

    # Save descriptors
    os.makedirs(os.path.dirname(SKY_FINDER_DESCRIPTORS_PATH), exist_ok=True)
    with open(SKY_FINDER_DESCRIPTORS_PATH, "w") as f:
        json.dump(descriptors, f, indent=4)

if __name__ == "__main__":
    main()