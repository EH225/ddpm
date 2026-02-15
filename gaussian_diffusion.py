"""
This module defines the GaussianDiffusion class which is used to generate images from noise and train the
U-Net model to run a 1-step denoising operation.
"""

import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, Tuple


class GaussianDiffusion(nn.Module):
    def __init__(self, model, *args, image_size: int, timesteps: int = 100, objective: str = "pred_noise",
                 beta_schedule: str = "sigmoid"):
        """
        Instantiates a Gaussian Diffusion Model instance.

        :param model: A torch model used to run iterative denoising steps on input noisey images.
        :param image_size: The height and width of input images, images are assumed to be square.
        :param timesteps: The number of iterative denoising timesteps to use to generate an image i.e. to
            fully decode a pure Gaussian noise start to a clean x_0 image.
        :param objective: Either pred_noise or pred_x_start which defines the training objective of the model
            and what it produces i.e. either the estimated noise added to the original image or the original
            image itself less the noise.
        :param beta_schedule: A beta schedule to use for training i.e. either linear, cosine, or sigmoid.
            This controls the levels of noise used most often during training at each time step. If set to
            linear, then the level of noise increase linearly at each timestep. If cosine, the noise increases
            in a way that is a smooth and gradual increase which can be better suited for DDPM training.
            Sigmoid increases the level of noise more sharply during the middle of the process, this can
            increase the speed of training but may be more prone to instability or sharp transitions.
        """
        super().__init__()
        self.model = model  # A model to use for the denoising steps e.g. a U-Net instance
        self.channels = 3  # The number of input image channels
        self.image_size = image_size  # The height and width, images are expected to be square
        self.objective = objective  # Specify which objective the model is to predict

        objectives = ["pred_noise", "pred_x_start"]
        assert objective in objectives, f"objective must be one of: {objectives}"

        # Helpful constants are registered below as buffers and can be access through self.name
        # This ensures that they are on the same device as the model parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for details
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        ### Noise schedule and alpha values
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        ### Add in other coefficients needed to transform between x_t, x_0 and noise:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise, where noise is sampled from N(0, 1)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        ### Add coeffs for posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the DDPM paper
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer("posterior_mean_coef1",
                        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2",
                        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        ### Add weights for the loss calculation
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Maps values between [0, 1] to values [-1, 1].
        """
        return img * 2 - 1

    def unnormalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Maps values between [-1, 1] to values [0, 1].
        """
        return (img + 1) * 0.5

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
                                 ) -> torch.Tensor:
        """
        Computes x_0 (the original starting image) from x_t (the original imge corrupted with noise) given
        the time step t and the noise added.

        :param x_t: A batch of noise images of shape (batch_size, C, H, W).
        :param t: The time step of each image in the batch of size (batch_size, ).
        :param noise: A batch of Gaussian noise sampled from N(0, 1) of the same shape as x_t.
        :returns: x_0 the batch of starting images corresponding to the batch of noise images, x_t.
        """
        # Transform x_t and noise to get x_0
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # Rearrange the equation: x_t = sqrt(alpha_t)*x_0 + sqrt(1 - alpha_t)*eps
        # where x_0 = x_start, eps = noise
        x_0 = (x_t - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod
        return x_0

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor
                                 ) -> torch.Tensor:
        """
        Computes the noise implied by x_t (the original imge corrupted with noise) and x_0 (the original
        starting image) and time step.

        :param x_t: A batch of noise images of shape (batch_size, C, H, W).
        :param t: The time step of each image in the batch of size (batch_size, ).
        :param x_0: A batch of original images of shape (batch_size, C, H, W).
        :returns: A batch of noise that was added to x_0 to get x_t the same shape as x_t and x_0.
        """
        # Transform x_t and x_0 to get the noise term
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # Rearrange the equation: x_t = sqrt(alpha_t)*x_0 + sqrt(1 - alpha_t)*eps
        # where x_0 = x_start, eps = noise
        pred_noise = (x_t - sqrt_alphas_cumprod * x_0) / sqrt_one_minus_alphas_cumprod
        return pred_noise

    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the posterior mean and stddev i.e. q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of
        the DDPM paper.

        :param x_0: A batch of original images of shape (batch_size, C, H, W).
        :param x_t: A batch of noise images of shape (batch_size, C, H, W).
        :param t: The time step of each image in the batch of size (batch_size, ).
        :returns:
            posterior_mean: (batch_size, C, H, W) tensor. Mean of the posterior.
            posterior_std: (batch_size, C, H, W) tensor. Std of the posterior.
        """
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_0 + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, model_kwargs: Dict = None) -> torch.Tensor:
        """
        Samples from p(x_{t-1} | x_t) according to Eq. (6) of the DDPM paper. This returns 1 step forward
        of the de-noising process i.e. x_{t-1} is 1 step less noisy than x_t with x_0 being a clean image.

        :param x_t: A batch of noise images of shape (batch_size, C, H, W).
        :param t: An integer denoting the denoising timestep currently being run. Note this is a single int
            and not a tensor of ints, it's the same int used for all images in the batch.
        :param model_kwargs: A dictionary of additional model inputs including "text_emb" for a possible
            test embedding of shape (batch_size, condition_dim).
        :returns: A batch of images x_{t-1} that are 1 step less noisy, same size and shape as x_t.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs

        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (b,) of all the same val t
        # sample x_{t-1} from p(x_{t-1} | x_t)
        # Get the model's prediction, note the model can predict either x_0 or the noise
        if self.objective == "pred_x_start":  # The model output will be the predicted x_start
            x_0 = self.model(x_t, t, model_kwargs)
        elif self.objective == "pred_noise":  # The model output will be the predicted noise
            noise = self.model(x_t, t, model_kwargs)
            x_0 = self.predict_start_from_noise(x_t, t, noise)  # Convert to x_0
        x_0 = x_0.clamp(-1, 1)  # Clamp to the valid range [-1, 1] to ensure the generate remains stable

        # Get the mean and std for q(x_{t-1} | x_t, x_0) using self.q_posterior, and sample x_{t-1}
        posterior_mean, posterior_std = self.q_posterior(x_0, x_t, t)
        x_tm1 = posterior_mean + posterior_std * torch.randn_like(x_t)
        return x_tm1

    @torch.no_grad()
    def sample(self, batch_size: int = 16, return_all_timesteps: bool = False, model_kwargs: Dict = None
               ) -> torch.Tensor:
        """
        Generates a batch of generated images of size (batch_size, C, H, W) given the input model_kwargs
        which will provide optional text embedding context. This method begins with batch_size Gaussian
        pure noise images of size (C, H, W) and applies a series of iterative denoising operations to them
        and returns the clean images when finished with values [0, 1].

        :param batch_size: The number of images to generate.
        :param return_all_timesteps: If set to True, then the first (all noise) and all T denoising timestep
            images are returned as a tensor of size (batch_size, T+1, C, H, W). Otherwise, just the last image
            is returned i.e. the maximally denoised one of size (batch_size, C, H, W).
        :param model_kwargs: A dictionary of additional model inputs including "text_emb" for a possible
            test embedding of shape (batch_size, condition_dim).
        :returns: A tensor of denoised image of size either:
                (batch_size, T+1, C, H, W) if return_all_timesteps is True else (batch_size, C, H, W)
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        self.eval()  # Set to eval mode for inference, switch off dropout and effects batch norm

        shape = (batch_size, self.channels, self.image_size, self.image_size)  # (B, C, H, W)
        img = torch.randn(shape, device=self.betas.device)  # Generate Gaussian noise ~ N(0, 1)
        imgs = [img]  # Create a list to hold the images that are denoised, starting with a pure noise image

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step",
                      total=self.num_timesteps):
            # Iteratively apply denoising steps to the image to move towards an "original", clean image
            imgs.append(self.p_sample(imgs[-1], t, model_kwargs=model_kwargs))

        res = imgs[-1] if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)  # Res has values [-1, 1] due to clamping, map to [0, 1] instead
        self.train()  # Return back to training mode afterwards
        return res

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Samples from q(x_t | x_0) according to Eq. (4) of the DDPM paper. This creates a noise image from a
        clean one i.e. x_0 by adding noise to it.

        :param x_0: A batch of original images of shape (batch_size, C, H, W).
        :param t: The time step of each image in the batch of size (batch_size, ).
        :param noise: A batch of Gaussian noise sampled from N(0, 1) of the same shape as x_0.
        :returns: A batch of noisy images that are a blend of x_0 clean images and Gaussian noise.
        """
        # q(x_t | x_0) = N(x_t; sqrt(alpha_t)*x_0, (1 - alpha_t)I)
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # Sampling from N(mu, sigma^2) can be done as: x_t = mu + sigma * noise where noise ~ N(0, 1)
        mu, sigma = sqrt_alphas_cumprod * x_0, sqrt_one_minus_alphas_cumprod
        x_t = mu + sigma * noise  # torch.randn_like(x_start)
        return x_t

    def p_losses(self, x_0: torch.Tensor, model_kwargs: Dict = None) -> torch.Tensor:
        """
        Computes a loss value for training using an input set of original, clean images x_0 and a dictionary
        of model_kwargs that can provide text embeddings as context. The following process is used:
            1). Randomly sample time steps t from [0, self.num_timesteps] for each x_0 image
            2). Randomly sample Gaussian noise for each x_0 image
            3). Generate an x_t using x_0 and the noise for each obs in the batch
            4). Pass x_t and t into the model and predict either the noise that was added or x_0
            5). Compare the model's prediction (i.e. the U-Net output) vs the ground truth

        :param x_0: A batch of original images of shape (batch_size, C, H, W).
        :param model_kwargs: A dictionary of additional model inputs including "text_emb" for a possible
            test embedding of shape (batch_size, condition_dim).
        :returns: A single torch float representing the loss from running a training iteration for 1 batch.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        b, nts = x_0.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_0.device).long()  # (b,) batch of random timesteps
        x_0 = self.normalize(x_0)  # (b, C, H, W) convert [0, 1] to [-1, 1] values
        noise = torch.randn_like(x_0)  # (b, C, H, W) create Gaussian noise N(0, 1) of the same shape
        target = noise if self.objective == "pred_noise" else x_0  # (b, C, H, W)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (b, C, H, W)
        # Implements the loss function according to Eq. (14) of the DDPM paper
        # Sample x_t from q(x_t | x_0) using the `q_sample` function
        x_t = self.q_sample(x_0, t, noise)  # Generate a noisy image using the starting image
        # Compute the y-hat values, will either be x_0 or noise, but will match target from above
        y_hat = self.model(x_t, t, model_kwargs)
        # Convert to FP32 before computing the loss
        # y_hat, target, loss_weight = y_hat.float(), target.float(), loss_weight.float()
        loss = (torch.pow(target - y_hat, 2) * loss_weight).mean()  # Compute the weighted MSE Loss
        return loss


########################
### Helper Functions ###
########################

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
    """
    Extracts the appropriate coefficient values based on the given timesteps.

    This function gathers the values from the coefficient tensor `a` according to the given timesteps `t`
    and reshapes them to match the required shape such that it supports broadcasting with the tensor of
    given shape `x_shape`.

    :param a: A tensor of shape (T,), containing coefficient values for all timesteps.
    :param t: A tensor of shape (b,), representing the timesteps for each sample in the batch.
    :param x_shape: The shape of the input image tensor, usually (B, C, H, W).
    :returns: A tensor of shape (B, 1, 1, 1), containing the extracted coefficient values from a for
        corresponding timestep of each batch element, reshaped accordingly.
    """
    b, *_ = t.shape  # Extract batch size from the timestep tensor
    out = a.gather(-1, t)  # Gather the coefficient values from `a` based on `t`
    out = out.reshape(b, *((1,) * (len(x_shape) - 1)))  # Reshape to (b, 1, 1, 1) for broadcasting
    return out


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Computes a linear schedule of beta values proposed in original DDPM paper.

    :param timesteps: The total number of timesteps to create beta balues for.
    :returns: A torch.tensor of beta values, one for each timestep.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Computes a cosine schedule of beta values proposed in Improved Denoising Diffusion Probabilistic Models
    (https://openreview.net/forum?id=-NEXDKk8gZ).

    :param timesteps: The total number of timesteps to create beta balues for.
    :returns: A torch.tensor of beta values, one for each timestep.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Computes a sigmoid schedule of beta values proposed in Scalable Adaptive Computation for Iterative
    Generation (https://arxiv.org/abs/2212.11972). Figure 8 suggets that this schedule is better for images
    of size 64 x 64 during training.

    :param timesteps: The total number of timesteps to create beta balues for.
    :returns: A torch.tensor of beta values, one for each timestep.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule: str, timesteps: int) -> torch.Tensor:
    """
    Computes a schedule of beta values, one for each timestep and returns them as a torch.Tensor.

    :param beta_schedule: Specifies which type of schedule to use i.e. linear, cosine, or sigmoid.
    :param timesteps: The number of iterative denoising timesteps to use to generate an image i.e. to
        fully decode a pure Gaussian noise start to a clean x_0 image.
    :returns: A torch.tensor of beta values, one for each timestep.
    """
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas
