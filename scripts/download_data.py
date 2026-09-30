"""Script to acquire raw dataset."""
from pathlib import Path
from object_detection.config.configuration import ConfigManager
from object_detection.data.download import generate_industrial_safety_dataset
from object_detection.utils.logger import setup_logger

logger = setup_logger("download_data")


def main():
    config = ConfigManager("configs/config.yaml")
    raw_dir = Path(config.dataset["raw_dir"])

    logger.info("Step 1: Downloading / Generating Raw Dataset...")
    dataset_path = generate_industrial_safety_dataset(raw_dir, num_samples=150)
    logger.info(f"Raw dataset ready at {dataset_path}")


if __name__ == "__main__":
    main()
