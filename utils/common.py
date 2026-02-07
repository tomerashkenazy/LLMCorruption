"""Common helpers for reproducibility and device selection."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: Optional[str] = None) -> str:
    """Pick a device string based on availability and request."""
    if requested:
        return requested
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
