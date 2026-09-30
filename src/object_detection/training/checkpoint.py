"""Model Checkpoint Manager for saving and loading training states."""
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import subprocess

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model state saving and loading with full experiment metadata."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        train_loss: float,
        val_loss: float,
        config: Dict[str, Any],
        filename: str = "best_model.pth"
    ) -> Path:
        """Saves model state and training parameters to disk.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            scheduler: PyTorch learning rate scheduler.
            epoch: Current epoch number.
            train_loss: Training loss value.
            val_loss: Validation loss value.
            config: Full experiment configuration.
            filename: Target checkpoint filename.

        Returns:
            Path: Path to saved checkpoint file.
        """
        checkpoint_path = self.save_dir / filename
        try:
            git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_revision = "unavailable"
        dataset_cfg = config.get("dataset", {})
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
            "metadata": {
                "architecture": config.get("model", {}).get("name", "fasterrcnn_resnet50_fpn"),
                "class_mapping": dataset_cfg.get("class_mapping", {}),
                "num_classes": dataset_cfg.get("num_classes"),
                "image_size": {"min_size": dataset_cfg.get("min_size"), "max_size": dataset_cfg.get("max_size")},
                "git_revision": git_revision,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path} (Epoch {epoch}, Val Loss: {val_loss:.4f})")
        return checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """Loads model weights and state from checkpoint file.

        Args:
            model: PyTorch model.
            optimizer: Optional optimizer to restore state.
            scheduler: Optional scheduler to restore state.
            checkpoint_path: Path to checkpoint file.
            device: Hardware device to map loaded tensors.

        Returns:
            Dict[str, Any]: Loaded checkpoint metadata dictionary.
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / "best_model.pth"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Successfully loaded checkpoint from {checkpoint_path} (Epoch {checkpoint.get('epoch', 0)})")
        return checkpoint
