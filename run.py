"""
This script is used to train the U-Net based diffusion model.
"""
import sys, os, argparse

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

from emoji_dataset import EmojiDataset
from unet import UNet
from gaussian_diffusion import GaussianDiffusion
from ddpm_trainer import Trainer
from typing import Dict


def train_model(config: Dict) -> None:
    """
    Runs training for the model using the configurations specified in the config file which can contain
    configurations for the U-Net, the GaussianDiffusion model, and the Trainer objects.
    """
    # 1). Init the U-Net model that is able to make iterative denoising step predictions
    unet_model = UNet(**config.get("UNet", {}))
    print("Number of parameters:", sum(p.numel() for p in unet_model.parameters()))

    # 2).Init the Gaussian Diffusion model with the U-Net as its de-noising model
    ddpm = GaussianDiffusion(unet_model, **config.get("GaussianDiffusion", {}))

    # 3). Build the EmojiDataset to train on
    dataset = EmojiDataset(ddpm.image_size)

    # 4). Configure the training pipeline
    trainer = Trainer(ddpm, dataset, **config.get("Trainer", {}))

    # 5). Train the model to completion
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Module")
    parser.add_argument("--debug", help="Set to True to run in debug mode")
    args = parser.parse_args()
    debug = args.debug.lower() == "true"

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
            "image_size": 64,
            "timesteps": 100,
            "objective": "pred_noise",
            "beta_schedule": "cosine",
        },
        "Trainer": {
            "batch_size": 256,
            "lr_start": 1e-3,
            "lr_end": 1e-6,
            "weight_decay": 0.0,
            "train_num_steps": 200000,
            "warm_up_pct": 0.05,
            "sample_every": 1000,
            "save_every": 5000,
            "results_folder": f"{CURRENT_DIR}/results",
            "use_amp": True,
        }
    }

    if debug: # If run in debug mode, shorten the training
        config["GaussianDiffusion"]["timesteps"] = 5
        config["Trainer"]["batch_size"] = 16
        config["Trainer"]["train_num_steps"] = 100
        config["Trainer"]["sample_every"] = 10
        config["Trainer"]["save_every"] = 25
        config["Trainer"]["results_folder"] = f"{CURRENT_DIR}/debug"

    train_model(config)
