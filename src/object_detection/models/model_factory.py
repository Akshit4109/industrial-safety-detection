"""Model Factory for creating and initializing Object Detection models."""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from object_detection.models.faster_rcnn import build_faster_rcnn_model
from object_detection.utils.device import get_device
from object_detection.utils.logger import setup_logger

logger = setup_logger("model_factory")


class ModelFactory:
    """Factory class to create object detection model instances based on configuration."""

    @staticmethod
    def create_model(
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Instantiates model according to configuration dictionary.

        Args:
            config: Full or sectioned project configuration dictionary.
            device: Target torch device. If None, auto-detected via get_device().

        Returns:
            nn.Module: Model placed on target hardware device.
        """
        model_config = config.get("model", config)
        dataset_config = config.get("dataset", config)

        model_name = model_config.get("name", "fasterrcnn_resnet50_fpn")
        num_classes = dataset_config.get("num_classes", 5)
        pretrained = model_config.get("pretrained", True)
        trainable_backbone_layers = model_config.get("trainable_backbone_layers", 3)
        box_score_thresh = model_config.get("box_score_thresh", 0.5)
        box_nms_thresh = model_config.get("box_nms_thresh", 0.5)
        min_size = dataset_config.get("min_size", 640)
        max_size = dataset_config.get("max_size", 768)

        if device is None:
            device = get_device()

        if model_name == "fasterrcnn_resnet50_fpn":
            model = build_faster_rcnn_model(
                num_classes=num_classes,
                pretrained=pretrained,
                trainable_backbone_layers=trainable_backbone_layers,
                box_score_thresh=box_score_thresh,
                box_nms_thresh=box_nms_thresh,
                min_size=min_size,
                max_size=max_size,
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

        model.to(device)
        logger.info(f"Model successfully loaded and transferred to device: {device}")
        return model
