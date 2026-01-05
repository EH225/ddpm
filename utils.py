"""
This module contains general utility functions.
"""
import torch


def get_device():
    """
    Auto-detects what hardware is available and returns the appropriate device.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_amp_dtype(device: str = "cuda"):
    """
    Determines the Automatic Mixed Precision data type that can be used on the current hardware.
    """
    assert isinstance(device, str), "device must be a str"
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float16

    # Get compute capability (major, minor)
    major, minor = torch.cuda.get_device_capability()

    # Ampere (8.x), Hopper (9.x), Ada (8.9) → BF16 supported
    bf16_supported = (major >= 8)

    return torch.bfloat16 if bf16_supported else torch.float16
