# Denoising Diffusion Probabilistic Model for Emoji Image Generation Project
This repository contains code for the Denoising Diffusion Probabilistic Model for Emoji Image Generation Project which utilizes a U-Net CNN deep learning model to perform generative, text-conditioned image sampling. Below is a quick overview of the repo layout:
- `emoji_dataset.py`: This folder contains the source code used for creating and loading in a pre-processed dataset used for training.
- `unet.py`: This module contains the source code of the U-Net deep neural network, the model used to make iterative denoising steps.
- `gaussian_diffusion.py`: This module contains the source code for performing training and sampling according to the DDPM framework.
- `utils.py`: This file contains helper functions used throughout the project.
- `ddpm_trainer.py`: This module contains code used to run the pytorch training loop.
- `run.py`: This is the main driver script of the project and is used to train diffusion model i.e. `python run.py`.
- `env_req`: This folder contains `environment.yml` and `environment_cuda.yml` which are files denoting the configuration of the virtual environment required to run the modules of this project.

This project leveraged materials from Stanford University's Deep Learning for Computer Vision ([XCS231N](https://cs231n.stanford.edu/)) course, with many modifications.