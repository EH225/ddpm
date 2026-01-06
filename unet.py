"""
This module defines the U-Net CNN model used to perform the iterative denoising steps.
"""

import copy, math
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict

def Upsample(dim_in: int, dim_out: int):
    """
    A Conv2d block that up-samples the image feature resolution a factor of 2.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim_in, dim_out, 3, padding=1),
    )


def Downsample(dim_in: int, dim_out: int) -> nn.Conv2d:
    """
    A nn.Conv2d block that down-samples the image feature resolution a factor of 2.
    """
    return nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)


class RMSNorm(nn.Module):
    """
    RMSNorm layer which is compute-efficient simplified variant of LayerNorm.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal position embedding for time steps.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlock(nn.Module):
    """
    A convolution block with feature modulation.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, scale_shift: torch.Tensor = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        # Scale and shift are used to modulate the output. This is a variant of feature fusion,
        # more powerful than simply adding the feature maps.
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    """
    A ResNet-like block with context dependent feature modulation.
    """

    def __init__(self, dim_in: int, dim_out: int, context_dim: int, drop_prob: float = 0.1):
        super().__init__()
        self.dim_in = dim_in  # The number of channels coming in
        self.dim_out = dim_out  # The number of channels going out
        self.context_dim = context_dim  # The size of the context vector
        self.drop_prob = drop_prob  # Record the prob used for dropout

        if context_dim is not None:
            self.mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(context_dim, dim_out * 2))
        else:
            self.mlp = None

        self.block1 = ConvBlock(dim_in, dim_out)
        self.block2 = ConvBlock(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and context is not None:
            context = self.mlp(context)
            context = rearrange(context, "b c -> b c 1 1")
            scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    """
    U-Net CNN model definition consisting of:
        1). An initial Conv2d layer
        2). Multiple down-blocks each consisting of: [ResnetBlock, ResnetBlock, Downsample]
        3). 2 ResnetBlock middle blocks
        4). The same number of up-blocks each consisting of: [Upsample, ResnetBlock, ResnetBlock] with skip
            connections to the earlier down-block outputs
        5). A final Conv2d layer producing an image of the same dimension as the input
    """

    def __init__(self, dim: int = 48, condition_dim: int = 512, dim_mults: Tuple[int] = (1, 2, 4, 8),
                  channels: int = 3, uncond_prob: float = 0.2):
        """
        Instantiates a U-Net model.

        :param dim: The number of channels at the first layer in the U-Net model.
        :param condition_dim: The dimension of the conditional vector (timestap + input text info).
        :param dim_mults: This parameter determines the number of conv blocks in both the encoder and decoder
            portions of the U-Net and details how many channels will be in each where each int is a multiple
            of the original dim channels used in the first and last layers of the network.
        :param channels: The number of channels of the input images and output images.
        :param uncond_prob: Probability of dropping the condition context vector during training.
        """
        super().__init__()
        self.channels = channels  # Record how many channels for the final output image leaving the model

        # Record the number of channels at each conv block layer i.e. [d1, d2, ..., dn], starts with dim
        # for the initial conv layer and then the inner blocks are specified as mults of dim
        dims = [dim] + [dim * m for m in dim_mults]

        # Store the input and output channel counts for each U-Net block in the down-sampling layers
        # e.g. [(d1, d2), (d2, d3), ..., (dn-1, dn)]
        in_out_downs = list(zip(dims[:-1], dims[1:]))
        # Store the input and output channel counts for each U-Net block in the up-sampling layers
        # e.g. [(dn, dn-1), (dn-1, dn-2), ..., (d2, d1)]
        in_out_ups = [(b, a) for a, b in reversed(in_out_downs)]

        # Encode the denoising timestep as a context vector of size dim * 4
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Encode the text condition (i.e. the text embedding from CLIP) as a context vector
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Record the probability of dropping the conditioning vector during training
        self.uncond_prob = uncond_prob

        # U-Net down-sampling and up-sampling blocks as a ModuleList of ModuleLists
        self.downs, self.ups = nn.ModuleList([]), nn.ModuleList([])

        ######################### Define Model Architecture #########################

        # 1). An initial convolutional layer which goes from channels to dim
        self.init_conv = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=3, padding=1)

        # 2). Create the down-sampling blocks, each down-sampling layer i.e. a down_block is a ModuleList
        # comprised of 3 blocks: [ResnetBlock, ResnetBlock, Downsample] which operates on dim_in channels
        # and outputs dim_out channels. context_dim is also provided to pass in the context vector.
        for idx, (dim_in, dim_out) in enumerate(in_out_downs):
            down_block = nn.ModuleList([ResnetBlock(dim_in, dim_in, context_dim),
                                        ResnetBlock(dim_in, dim_in, context_dim),
                                        Downsample(dim_in, dim_out)])
            self.downs.append(down_block)

        # 3). Create 2 middle ResNet blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        # 4). Create the up-sampling blocks, each up-sampling layer i.e. an up_block is a ModuleList comprised
        # of 3 blocks: [Upsample, ResnetBlock, ResnetBlock] which operates on dim_in channels and outputs
        # dim_out channels. context_dim is also provided to pass in the context vector.
        for idx, (dim_in, dim_out) in enumerate(in_out_ups):
            # To account for the skip connections coming from the encoder down-blocks, the input channels
            # here are actually x2 so that we can concat the outputs from earlier in the network
            up_block = nn.ModuleList([Upsample(dim_in, dim_out),
                                      ResnetBlock(dim_out * 2, dim_out, context_dim),
                                      ResnetBlock(dim_out * 2, dim_out, context_dim)])

            self.ups.append(up_block)

        # 5). Add 1 final final convolution to map to the output channels
        self.final_conv = nn.Conv2d(in_channels=dim, out_channels=channels, kernel_size=1)

    def cfg_forward(self, x: torch.Tensor, time: torch.Tensor, model_kwargs: Dict = None) -> torch.Tensor:
        """
        Classifier-free guidance forward pass method. model_kwargs should contain `cfg_scale`. An output
        image is produced using the context provided and without, and then the 2 are combined together:

            x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
            where eps is the UNet model forward pass.

        :param x: An input tensor of shape (batch_size, channels, height, width).
        :param time: An input tensor of shape (batch_size, ) containing the timesteps of each image.
        :param model_kwargs: A dictionary of additional model inputs including "text_emb" for a possible
            test embedding of shape (batch_size, condition_dim).
        :returns: An output tensor of shape (batch_size, channels, height, width) matching the input x.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        cfg_scale = model_kwargs["cfg_scale"]
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)  # Copy to avoid mutation below
        # Apply classifier-free guidance using:
        #   x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
        x1 = self.forward(x, time, model_kwargs)  # Generate the output x1 with the context
        model_kwargs["text_emb"] = None  # For unconditional sampling, set text_emb to None
        x2 = self.forward(x, time, model_kwargs)  # Generate again without the context
        x = (cfg_scale + 1) * x1 - cfg_scale * x2  # Combine together into 1 output image
        return x

    def forward(self, x: torch.Tensor, time: torch.Tensor, model_kwargs: Dict = None) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        :param x: An input tensor of shape (batch_size, channels, height, width).
        :param time: An input tensor of shape (batch_size, ) containing the timesteps of each image.
        :param model_kwargs: A dictionary of additional model inputs including "text_emb" for a possible
            test embedding of shape (batch_size, condition_dim).
        :returns: An output tensor of shape (batch_size, channels, height, width) matching the input x.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs

        if "cfg_scale" in model_kwargs:  # If specified, then run using the classifier-free guidance method
            return self.cfg_forward(x, time, model_kwargs)

        context = self.time_mlp(time)  # Embed the time step t as a deep vector representation

        # Embed condition and add to context
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:  # Default to a vector of zeros if no text embedding provided
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)

        if self.training:  # Randomly drop condition i.e. the input text provided to guide image generation
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
            mask = mask[:, None].to(cond_emb.device)  # (batch_size, 1)
            cond_emb = cond_emb * mask  # Randomly zero out the text embs of this batch with p=uncond_prob
        context = context + self.condition_mlp(cond_emb)  # Timestep and text input as a context vec combined

        # Process the input batch of images x through the U-Net conditioned on the context vector
        # 1). Initial convolution
        x = self.init_conv(x)

        # 2). Pass the intermediate x through the down-blocks
        resid_conn_features = []  # Use a stack for FIFO processing of the residual connection feature maps
        for down_block in self.downs:  # Iterate over all the down-blocks, which each have 3 components
            x = down_block[0](x, context)  # ResnetBlock 1
            resid_conn_features.append(x)  # Record this layer's outputs for the residual connection
            x = down_block[1](x, context)  # ResnetBlock 2
            resid_conn_features.append(x)  # Record this layer's outputs for the residual connection
            x = down_block[2](x)  # Downsample block

        # 3). Pass the intermediate x through the mid-blocks
        x = self.mid_block1(x, context)
        x = self.mid_block2(x, context)

        # 4). Pass the intermediate x through the up-blocks
        for up_block in self.ups:  # Iterate over all the up-blocks, which each have 3 components
            x = up_block[0](x)  # Upsample block
            x = up_block[1](torch.concat([x, resid_conn_features.pop()], dim=1), context)  # ResnetBlock 1
            x = up_block[2](torch.concat([x, resid_conn_features.pop()], dim=1), context)  # ResnetBlock 2

        # 5). Final conv block
        x = self.final_conv(x)

        return x
