"""Training Engine for Faster R-CNN with TensorBoard experiment logging."""
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

from object_detection.training.checkpoint import CheckpointManager
from object_detection.utils.logger import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """Trains and validates Faster R-CNN object detection models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 5)
        self.lr = train_cfg.get("learning_rate", 0.005)
        self.momentum = train_cfg.get("momentum", 0.9)
        self.weight_decay = train_cfg.get("weight_decay", 0.0005)
        self.accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        self.max_gradient_norm = float(train_cfg.get("max_gradient_norm", 0.0))
        self.mps_empty_cache_interval = max(0, int(train_cfg.get("mps_empty_cache_interval", 0)))

        # Optimizer: SGD with momentum and weight decay for Faster R-CNN
        params = [p for p in model.parameters() if p.requires_grad]
        self.trainable_params = params
        self.optimizer = optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # Learning Rate Scheduler: StepLR
        step_size = train_cfg.get("lr_step_size", 3)
        gamma = train_cfg.get("lr_gamma", 0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # Checkpoint and Loggers
        save_dir = Path(train_cfg.get("save_dir", "artifacts/models"))
        log_dir = Path(train_cfg.get("log_dir", "runs"))
        self.checkpoint_manager = CheckpointManager(save_dir)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        self.best_val_loss = float("inf")

    def should_optimizer_step(self, batch_index: int, num_batches: int) -> bool:
        """Return whether an accumulation window ends at this zero-based batch."""
        return (batch_index + 1) % self.accumulation_steps == 0 or batch_index + 1 == num_batches

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs a single training epoch.

        Returns:
            Dict[str, float]: Dictionary of average training losses.
        """
        self.model.train()
        running_losses = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "total_loss": 0.0
        }

        num_batches = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite training loss at batch {batch_idx + 1}: {total_loss.item()}")

            # Divide only the backpropagated loss. Metrics retain the true loss.
            (total_loss / self.accumulation_steps).backward()
            is_step = self.should_optimizer_step(batch_idx, num_batches)
            if is_step:
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_gradient_norm) if self.max_gradient_norm > 0 else torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in self.trainable_params if p.grad is not None]))
                if not torch.isfinite(gradient_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"Non-finite gradient norm at batch {batch_idx + 1}: {gradient_norm.item()}")
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                logger.debug("Optimizer step at batch %s; pre-clip gradient norm %.4f", batch_idx + 1, gradient_norm.item())

            # Accumulate loss metrics
            for k, v in loss_dict.items():
                running_losses[k] += v.item()
            running_losses["total_loss"] += total_loss.item()

            # Explicitly drop batch graph references before the next MPS batch.
            del loss_dict, total_loss, images, targets
            if self.device.type == "mps" and self.mps_empty_cache_interval and is_step and (batch_idx + 1) % self.mps_empty_cache_interval == 0:
                torch.mps.empty_cache()

        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        return avg_losses

    def validate_epoch(self) -> Dict[str, float]:
        """Runs validation loss evaluation across validation set.

        Returns:
            Dict[str, float]: Dictionary of average validation losses.
        """
        self.model.train()  # Keep in train mode to compute loss values without updating weights
        running_losses = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "total_loss": 0.0
        }

        num_batches = len(self.val_loader)
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

                for k, v in loss_dict.items():
                    running_losses[k] += v.item()
                running_losses["total_loss"] += total_loss.item()
                del loss_dict, total_loss, images, targets

        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        return avg_losses

    def fit(self) -> Dict[str, Any]:
        """Runs full training pipeline across configured epochs.

        Returns:
            Dict[str, Any]: Training summary metrics.
        """
        logger.info(f"Starting Training for {self.epochs} Epochs on Device: {self.device}")

        for epoch in range(1, self.epochs + 1):
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Logging to TensorBoard
            self.writer.add_scalar("Loss/train_total", train_losses["total_loss"], epoch)
            self.writer.add_scalar("Loss/train_classifier", train_losses["loss_classifier"], epoch)
            self.writer.add_scalar("Loss/train_box_reg", train_losses["loss_box_reg"], epoch)
            self.writer.add_scalar("Loss/val_total", val_losses["total_loss"], epoch)
            self.writer.add_scalar("Loss/val_classifier", val_losses["loss_classifier"], epoch)
            self.writer.add_scalar("Loss/val_box_reg", val_losses["loss_box_reg"], epoch)
            self.writer.add_scalar("Params/learning_rate", current_lr, epoch)

            logger.info(
                f"Epoch [{epoch}/{self.epochs}] | "
                f"Train Loss: {train_losses['total_loss']:.4f} | "
                f"Val Loss: {val_losses['total_loss']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint best model
            if val_losses["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_losses["total_loss"]
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_loss=train_losses["total_loss"],
                    val_loss=val_losses["total_loss"],
                    config=self.config,
                    filename="best_model.pth"
                )

            # Save last model checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                train_loss=train_losses["total_loss"],
                val_loss=val_losses["total_loss"],
                config=self.config,
                filename="last_model.pth"
            )

        self.writer.close()
        logger.info("Training pipeline completed successfully.")
        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.epochs
        }
