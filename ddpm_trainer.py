import math, os
import torch
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import Tuple, List


def cycle(dl):
    """
    Generator to cycle through batches of the data loader.
    """
    while True:
        for data in dl:
            yield data


class Trainer:
    def __init__(self, diffusion_model, dataset, device: str, *args, train_batch_size: int = 256,
                 train_lr: float = 1e-3, weight_decay: float = 0.0, train_num_steps: int = 100000,
                 adam_betas: Tuple[float] = (0.9, 0.99), sample_every: int = 1000, save_every: int = 5000,
                 results_folder: str = None):
        """
        A framework for loading, saving, and training a diffusion model.

        :param diffusion_model: A torch model to be trained.
        :param dataset: A data set used for training.
        :param device: The device to train on e.g. cpu or cuda.
        :param train_batch_size: The batch size to use during training.
        :param train_lr: # The training learning rate.
        :param weight_decay: The weight_decay to provide to the Adam optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param sample_every: An int denoting how often to sample and save outputs from the model.
        :param save_every: An int denoting how often to save the model weights.
        :param results_folder: A location to save the result of the training.
        """
        super().__init__()

        assert results_folder is not None, "must specify results folder"
        self.diffusion_model = diffusion_model

        self.device = device
        self.num_samples = 25  # The number of samples to generate periodically and save from the model
        self.save_every = save_every
        self.sample_every = sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        # Set the dataset and dataloader
        self.ds = dataset
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=0)
        self.dl = cycle(dl)

        # Configure the optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas,
                        weight_decay=weight_decay)

        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)  # Create the directory if not already there

        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        os.makedirs(self.checkpoints_folder, exist_ok=True)

        self.samples_folder = os.path.join(self.results_folder, "samples/")
        os.makedirs(self.samples_folder, exist_ok=True)

        self.step = 0  # Training step counter
        self.all_losses = []  # Aggregate loss values during training

    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weightes were saved.
        :returns: None.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        print(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "all_losses": self.all_losses,
                "model": self.diffusion_model.state_dict(),
                "opt": self.opt.state_dict(),
                }
        torch.save(data, checkpoint_path)

    def load(self, milestone: int) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weightes were saved.
        :returns: None.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        print(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, loss values, model weights and optimizer state from the
        # checkpoint data read in from disk
        self.step = checkpoint_data["step"]
        self.all_losses = checkpoint_data.get("all_losses", [])
        self.diffusion_model.load_state_dict(checkpoint_data["model"])
        self.opt.load_state_dict(checkpoint_data["opt"])

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def train(self) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations.
        Returns a list of the losses obtained for each training timestep.
        """
        self.diffusion_model.to(self.device)  # Move the model to the correct device
        self.diffusion_model.train()

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                data, model_kwargs = next(self.dl)
                data = data.to(self.device)
                model_kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v
                                for k, v in model_kwargs.items()}

                self.opt.zero_grad()  # Zero the gradients of the optimizer before computing the loss
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    loss = self.diffusion_model.p_losses(data, model_kwargs=model_kwargs)
                loss.backward()  # Compute gradients wrt the parameters of the model
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)  # Apply clipping
                self.opt.step()  # Update the model parameters by taking a gradient step

                pbar.set_description(f"loss: {loss.item():.4f}")
                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep

                self.step += 1

                if self.step % self.save_every == 0:  # Periodically save the model weights to disk
                    self.save(self.step)

                if self.step % self.sample_every == 0:  # Periodically generate samples from the model
                    self.diffusion_model.eval()

                    with torch.no_grad():
                        model_kwargs = self.ds.random_model_kwargs(self.num_samples)
                        model_kwargs["text_emb"] = model_kwargs["text_emb"].to(self.device)

                        all_images = self.diffusion_model.sample(batch_size=self.num_samples,
                                                                 model_kwargs=model_kwargs)

                    save_image(all_images, os.path.join(self.samples_folder, f"sample-{self.step}.png"),
                               nrow=int(math.sqrt(self.num_samples)))
                    self.diffusion_model.train()  # Switch back to training mode once finished

                pbar.update(1)
