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
