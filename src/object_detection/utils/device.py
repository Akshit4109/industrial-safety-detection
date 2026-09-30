"""Configurable PyTorch device selection with safe hardware fallbacks."""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Return the requested available device, or choose CUDA, MPS, then CPU.

    ``preference`` may be ``auto``, ``cuda``, ``mps``, or ``cpu``. Explicit
    unavailable accelerators fall back to CPU and log why; they never silently
    claim that hardware is in use.

    Returns:
        torch.device: Active device.
    """
    preference = preference.lower()
    if preference not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device preference '{preference}'. Use auto, cuda, mps, or cpu.")
    if preference == "cpu":
        device = torch.device("cpu")
    elif preference in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    elif preference in {"auto", "mps"} and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: mps (Apple Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        if preference == "mps":
            logger.warning("MPS was requested but is unavailable; falling back to CPU.")
        elif preference == "cuda":
            logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
        else:
            logger.info("Using device: cpu (no CUDA or available MPS backend)")
    logger.info("Selected execution device: %s (preference=%s)", device, preference)
    return device
