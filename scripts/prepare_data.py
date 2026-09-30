"""Script to validate raw data and prepare split datasets."""
from pathlib import Path
from object_detection.config.configuration import ConfigManager
from object_detection.data.validation import DatasetValidator
from object_detection.data.preprocessing import prepare_and_split_dataset, visualize_sample_annotations
from object_detection.utils.logger import setup_logger

logger = setup_logger("prepare_data")


def main():
    config = ConfigManager("configs/config.yaml")
    raw_dir = Path(config.dataset["raw_dir"])
    processed_dir = Path(config.dataset["processed_dir"])
    valid_classes = [name for id_, name in config.dataset["class_mapping"].items() if name != "__background__"]

    logger.info("Step 2: Validating Raw Dataset...")
    validator = DatasetValidator(raw_dir=raw_dir, valid_classes=valid_classes)
    report = validator.validate_dataset()

    valid_samples = report["valid_samples"]
    if not valid_samples:
        raise ValueError(f"No valid dataset samples found in {raw_dir}! Run download_data.py first.")

    logger.info("Step 3: Splitting Dataset & Saving Processed Splits...")
    splits = prepare_and_split_dataset(
        valid_samples=valid_samples,
        output_dir=processed_dir,
        train_ratio=config.dataset["train_split"],
        val_ratio=config.dataset["val_split"],
        test_ratio=config.dataset["test_split"],
        seed=config.project["seed"]
    )

    # Save a sample visualization overlay
    sample_vis_path = Path("artifacts/predictions/sample_ground_truth.png")
    visualize_sample_annotations(valid_samples[0], sample_vis_path)
    logger.info("Dataset Preparation Complete!")


if __name__ == "__main__":
    main()
