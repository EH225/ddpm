"""
This module contains general utility functions.
"""
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_device():
    """
    Auto-detects what hardware is available and returns the appropriate device.

    :returns: A torch device denoting what device is available as a torch.device, not a string.
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

    :param device: The device currently available as a string e.g. "cpu" or "cuda".
    :returns: A torch float type for auto mixed precision training i.e. torch.dtype (torch.float16 or
    torch.bfloat16) if AMP should be used, otherwise None.
    """
    assert isinstance(device, str), "device must be a str"

    if device == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8: # Ampere (8.0+) and newer → prefer BF16
            return torch.bfloat16
        else:
            return torch.float16

    return None # No AMP on CPU

def plot_and_save_loss(loss_dir: str) -> None:
    """
    Combines all the data cached to a directory of loss value outputs and combines them together to create a
    loss plot which is then saved down in the same directory as well.

    :param loss_dir: A directory containing losses-{milestone}.csv files.
    :returns: None, generates a plot that is then saved to disk.
    """
    filenames = [x for x in os.listdir(loss_dir) if x.startswith("losses") and x.endswith(".csv")]
    if len(filenames) > 0:  # Otherwise do nothing
        all_losses = []
        milestones = [int(x.replace("losses-", "").replace(".csv", "")) for x in filenames]
        for m in sorted(milestones):
            df = pd.read_csv(os.path.join(loss_dir, f"losses-{m}.csv"), index_col=0)
            all_losses.extend(df.iloc[:, 0].tolist())
        all_losses = pd.Series(all_losses)  # Convert to a pd.Series for ease of use
        all_losses.index += 1  # Set the index to begin at 1
        # Create a plot and save it to the same directory
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(all_losses, zorder=3)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Training Step")
        ax.set_title("Training Loss")
        ax.grid(color="lightgray", zorder=-3)
        fig.savefig(os.path.join(loss_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
