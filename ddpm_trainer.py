import math, os, copy
import torch
import logging
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from typing import Tuple, List
from utils import get_device, get_amp_dtype, plot_and_save_loss
from emoji_dataset import EmojiDataset
from gaussian_diffusion import GaussianDiffusion


def cycle(dl):
    """
    Generator to cycle through batches of the data loader.
    """
    while True:
        for data in dl:
            yield data


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Trainer:
    def __init__(self, diffusion_model: GaussianDiffusion, dataset: EmojiDataset, batch_size: int = 256,
                 lr_start: float = 1e-4, lr_end: float = 1e-6, weight_decay: float = 0.0,
                 train_num_steps: int = 100000, warm_up_pct: float = 0.1,
                 adam_betas: Tuple[float] = (0.9, 0.999), grad_clip: float = 1.0,
                 sample_every: int = 1000, save_every: int = 5000, results_folder: str = None,
                 use_amp: bool = False, use_latest_checkpoint: bool = True):
        """
        A framework for loading, saving, and training a de-noising diffusion model.

        :param diffusion_model: A torch model to be trained by this trainer object.
        :param dataset: A data set used for training.
        :param batch_size: The batch size to use during training.
        :param lr_start: The initial learning rate (after warm up).
        :param lr_end: The terminal training learning rate.
        :param weight_decay: The weight_decay to provide to the AdamW optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param warm_up_pct: The percentage of train_num_steps over which the learning rate warm up period
            will be run i.e. ramps from very low to a peak of lr_start.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param sample_every: An int denoting how often to sample and save outputs from the model.
        :param save_every: An int denoting how often to save the model weights.
        :param results_folder: A location to save the result of the training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to True, then the latest checkpoint detected in the results
            directory will be loaded in before training begins to pick up from where it was last left off.
        """
        super().__init__()

        # 1). Create directories to save results
        assert results_folder is not None, "You must specify results folder to save the outputs"
        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        self.losses_folder = os.path.join(self.results_folder, "losses/")
        self.samples_folder = os.path.join(self.results_folder, "samples/")
        for directory in [self.results_folder, self.checkpoints_folder,
                          self.losses_folder, self.samples_folder]:
            os.makedirs(directory, exist_ok=True)  # Create the directory if not already there

        # 2). Set up logging during training
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  # Prevent duplicate handlers
            file_handler = logging.FileHandler(os.path.join(self.results_folder, "train.log"),
                                               encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # file_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(file_handler)

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # tqdm_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(tqdm_handler)
        self.logger.propagate = False

        # 3). Record input parameters
        self.diffusion_model = diffusion_model
        print("Number of parameters:", sum(p.numel() for p in diffusion_model.parameters()))
        # Maintain a model that is an EMA of model weights for stability during sampling
        self.ema_model = copy.deepcopy(self.diffusion_model)
        self.ema_model.requires_grad_(False)
        self.ema_decay = 0.999  # This controls the amount of weight that is placed on the prior ema model
        # weights i.e. (1 - ema_decay) is used as the weight on the most recent model weights after another
        # gradient update step has been applied to it

        self.device = get_device().type  # Auto-detect what device to use for training
        self.grad_clip = grad_clip # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device) if use_amp else None
        self.num_samples = 25  # The number of samples to generate periodically and save from the model
        self.save_every = save_every # Specifies how often to save the model's weights
        self.sample_every = sample_every # Specifies how often to generate and save samples
        self.batch_size = batch_size # Specifies the batch size used during training
        self.train_num_steps = train_num_steps # The total number of training steps that will be taken

        # 4). Configure the dataset and dataloader
        if self.device == "cuda":
            num_workers, pin_memory, persistent_workers, prefetch_factor = 4, True, True, 16
        else:
            num_workers, pin_memory, persistent_workers, prefetch_factor = 0, False, False, None

        self.ds = dataset
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=pin_memory, persistent_workers=persistent_workers,
                          prefetch_factor=prefetch_factor)
        self.dl = cycle(dl)

        # 5). Configure the optimizer, learning rate scheduler, training step counter, and losses list
        self.opt = AdamW(diffusion_model.parameters(), lr=lr_start, betas=adam_betas,
                         weight_decay=weight_decay)

        warmup_steps = int(train_num_steps * warm_up_pct)  # Slowly ramp up the LR from very low to peak
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Cosine annealing of the learning rate during the rest of training
        decay = CosineAnnealingLR(self.opt, T_max=train_num_steps - warmup_steps, eta_min=lr_end)
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        self.step = 0  # Training step counter
        self.all_losses = []  # Aggregate loss values during training between each checkpoint save

        # 6). Load in pre-trained weights if available in the results/checkpoints directory
        if use_latest_checkpoint:
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-","").replace(".pt", "")) for x in checkpoints])
                self.load(last_checkpoint)  # Load in the most recent milestone to continue training from

    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to: {checkpoint_path}")
        data = {"step": self.step,
                "model": self.diffusion_model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                }
        torch.save(data, checkpoint_path)
        # Save down all the loss values produced by model training since the last caching
        pd.Series(self.all_losses).to_csv(os.path.join(self.losses_folder, f"losses-{milestone}.csv"))

    def load(self, milestone: int) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Weights and other trainer state parameter values are loaded into memory.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, model weights, and optimizer state from the checkpoint data
        # read in from disk
        self.step = checkpoint_data["step"]
        self.diffusion_model.load_state_dict(checkpoint_data["model"])
        self.ema_model.load_state_dict(checkpoint_data["ema_model"])
        self.opt.load_state_dict(checkpoint_data["opt"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    @torch.no_grad()
    def update_ema(self) -> None:
        """
        Updates the weights of the ema_model using the current weights in model with a decay parameter
        that specifies how much weight to place on the existing ema_model weights.
        """
        for ema_param, param in zip(self.ema_model.parameters(), self.diffusion_model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def report_lr_wd(self):
        """
        Reports the learning rates and weight decay parameter values of the optimizer.
        """
        self.logger.info(f"Reporting learning rates and weight decay at step={self.step}")
        for i, group in enumerate(self.opt.param_groups):  # Report all learning rates
            self.logger.info((f"lr = {group['lr']:.2e}, wd = {group['weight_decay']:.2e}, "
                              f"count = {len(group['params'])}"))

    def train(self) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations.
        """
        msg = f"Starting Training, step={self.step}, device={self.device}, amp_dtype={self.amp_dtype}"
        self.logger.info(msg)
        self.diffusion_model.to(self.device)  # Move the model to the correct device
        self.diffusion_model.train() # Make sure it is in training mode
        self.ema_model.requires_grad_(False)

        scaler = None
        if self.amp_dtype == torch.float16: # Only want to use grad scaler if using float16 AMP
            scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                data, model_kwargs = next(self.dl) # Get the next batch of training data
                data = data.to(self.device, non_blocking=True) # Move the data to the same device as model
                model_kwargs = {k: v.to(self.device,non_blocking=True) if torch.is_tensor(v) else v
                                for k, v in model_kwargs.items()}

                self.opt.zero_grad(set_to_none=True)  # Zero the gradients of the opt before computing loss
                grad_norm = None # Prevent from being undefined if self.grad_clip is None
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                        loss = self.diffusion_model.p_losses(data, model_kwargs=model_kwargs)
                else:
                    loss = self.diffusion_model.p_losses(data, model_kwargs=model_kwargs)

                if self.amp_dtype == torch.float16:
                    scaler.scale(loss).backward() # Compute gradients wrt the parameters of the model
                    if self.grad_clip is not None:
                        scaler.unscale_(self.opt)  # Unscale before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(),
                                                                   self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    loss.backward() # Compute gradients wrt the parameters of the model
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(),
                                                                   self.grad_clip)
                    self.opt.step() # Update the model parameters by taking a gradient step

                self.scheduler.step()  # Update the learning rate scheduler
                self.update_ema() # Update the ema_model's weights after this gradient update

                pbar.set_description(f"loss: {loss.item():.4f}, grad_norm: {grad_norm:.4f}")
                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep

                self.step += 1

                # Periodically save the model weights to disk
                if self.step % self.save_every == 0 or self.step == self.train_num_steps:
                    self.save(self.step)
                    plot_and_save_loss(self.losses_folder)  # Generate a new plot of the training losses
                    self.all_losses = []  # Clear the list of losses after each save, store only the ones
                    # from the last save to the next save
                    torch.cuda.empty_cache()

                # Periodically generate samples from the model
                if self.step % self.sample_every == 0 or self.step == self.train_num_steps:
                    self.logger.info((f"loss={loss.item():.4f}, grad_norm={grad_norm:.3f}, step={self.step}"))
                    self.report_lr_wd()
                    self.diffusion_model.eval() # Switch to eval model for sampling

                    with torch.no_grad():
                        model_kwargs = self.ds.random_model_kwargs(self.num_samples)
                        model_kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v
                                        for k, v in model_kwargs.items()}
                        model_kwargs["cfg_scale"] = 3.0
                        if self.amp_dtype is not None:
                            with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                                all_images = self.ema_model.sample(batch_size=self.num_samples,
                                                                   model_kwargs=model_kwargs)
                        else:
                            all_images = self.ema_model.sample(batch_size=self.num_samples,
                                                               model_kwargs=model_kwargs)

                    save_image(all_images, os.path.join(self.samples_folder, f"sample-{self.step}.png"),
                               nrow=int(math.sqrt(self.num_samples)))
                    self.diffusion_model.train()  # Switch back to training mode once finished

                del data, model_kwargs, loss

                pbar.update(1)
