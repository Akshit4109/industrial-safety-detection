"""Environment Verification Script."""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("verify_env")


def verify():
    logger.info("=== Running Environment Verification ===")

    # 1. Python
    logger.info(f"Python Version: {sys.version.split()[0]}")

    # 2. PyTorch & Device
    import torch
    logger.info(f"PyTorch Version: {torch.__version__}")
    from object_detection.utils.device import get_device
    device = get_device()
    logger.info(f"Detected PyTorch Device: {device}")

    # 3. Torchvision
    import torchvision
    logger.info(f"Torchvision Version: {torchvision.__version__}")

    # 4. OpenCV
    import cv2
    logger.info(f"OpenCV Version: {cv2.__version__}")

    # 5. DVC
    import dvc
    logger.info(f"DVC Version: {dvc.__version__}")

    # 6. TensorBoard
    import tensorboard
    logger.info(f"TensorBoard Version: {tensorboard.__version__}")

    # 7. FastAPI
    import fastapi
    logger.info(f"FastAPI Version: {fastapi.__version__}")

    # 8. Project Imports
    from object_detection.config.configuration import ConfigManager
    config = ConfigManager("configs/config.yaml")
    logger.info(f"Project Config Loaded Successfully: {config.project.get('name')}")

    logger.info("=== Environment Verification PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    verify()
