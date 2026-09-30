"""Seed utility for reproducible experiments."""
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed}")
