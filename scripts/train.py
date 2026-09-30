"""CLI script to trigger model training pipeline."""
from pathlib import Path
from torch.utils.data import DataLoader

from object_detection.config.configuration import ConfigManager
from object_detection.data.dataset import ObjectDetectionDataset, collate_fn
from object_detection.models.model_factory import ModelFactory
from object_detection.training.trainer import Trainer
from object_detection.utils.device import get_device
from object_detection.utils.seed import set_seed
from object_detection.utils.logger import setup_logger

logger = setup_logger("train_script")


def main():
    logger.info("=== Initializing Object Detection Training ===")
    config = ConfigManager("configs/config.yaml")

    # Set seed for reproducibility
    seed = config.project.get("seed", 42)
    set_seed(seed)

    # Configurable auto selection: CUDA -> MPS -> CPU.
    device = get_device(config.training.get("device", "auto"))
    logger.info("Training will use selected device: %s", device)

    # Load Data
    processed_dir = Path(config.dataset["processed_dir"])
    train_json = processed_dir / "train" / "annotations.json"
    val_json = processed_dir / "val" / "annotations.json"

    class_mapping = config.dataset["class_mapping"]
    batch_size = config.training.get("batch_size", 4)
    num_workers = config.training.get("num_workers", 2)

    logger.info("Loading training and validation datasets...")
    train_dataset = ObjectDetectionDataset(train_json, class_mapping, is_train=True)
    val_dataset = ObjectDetectionDataset(val_json, class_mapping, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # Create Model
    logger.info("Instantiating Faster R-CNN model via ModelFactory...")
    model = ModelFactory.create_model(config.get_all(), device=device)

    # Initialize Trainer and Run
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.get_all(),
        device=device
    )

    results = trainer.fit()
    logger.info(f"Training completed. Best Validation Loss: {results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
