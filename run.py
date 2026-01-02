"""
This script is used to train the U-Net based diffusion model.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

from emoji_dataset import EmojiDataset
from unet import UNet
from gaussian_diffusion import GaussianDiffusion
from ddpm_trainer import Trainer, get_device
from typing import Dict


def train_model(config: Dict) -> None:
    """
    Runs training for the model using the configurations specified in the config file which can contain
    configurations for the U-Net, the GaussianDiffusion model, and the Trainer objects.
    """
    # 1). Init the U-Net model that is able to make iterative denoising step predictions
    unet_model = UNet(**config.get("UNet", {}))
    print("Number of parameters:", sum(p.numel() for p in unet_model.parameters()))

    # 2).Init the Gaussian Diffusion model with the U-Net as its model
    ddpm = GaussianDiffusion(**config.get("GaussianDiffusion", {}))

    # 3). Build the EmojiDataset to train on
    dataset = EmojiDataset(ddpm.image_size)

    # 4). Configure the training pipeline
    trainer = Trainer(ddpm, dataset, get_device(), **config.get("Trainer", {}))

    # 5). Train the model to completion
    trainer.train()


if __name__ == "__main__":
    # Set up a config for training, set parameters for each component
    config = {
        "UNet": {
            "dim": 48,
            "condition_dim": 512,
            "dim_mults": (1, 2, 4, 8),
            "channels": 3,
            "uncond_prob": 0.2,
        },
        "GaussianDiffusion": {
            "image_size": 32,
            "timesteps": 100,
            "objective": "pred_noise",
            "beta_schedule": "sigmoid",
        },
        "Trainer": {
            "train_batch_size": 256,
            "train_lr": 1e-3,
            "weight_decay": 0.0,
            "train_num_steps": 100000,
            "sample_every": 1000,
            "save_every": 10000,
            "results_folder": "/results"
        }
    }

    train_model(config)
