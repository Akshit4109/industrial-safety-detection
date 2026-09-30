import torch
import torch.nn as nn
import ssl
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import logging

# Handle macOS SSL certificate verification for PyTorch hub weight downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

logger = logging.getLogger(__name__)


def build_faster_rcnn_model(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    box_score_thresh: float = 0.5,
    box_nms_thresh: float = 0.5,
    min_size: int = 640,
    max_size: int = 768,
) -> FasterRCNN:
    """Builds Faster R-CNN ResNet-50-FPN model with custom classification head.

    Args:
        num_classes: Total number of target classes including background (0: __background__).
        pretrained: Whether to load COCO pretrained weights.
        trainable_backbone_layers: Number of trainable backbone layers (0 to 5).
        box_score_thresh: Confidence score threshold for inference box filtering.
        box_nms_thresh: IoU threshold for Non-Maximum Suppression (NMS).

    Returns:
        FasterRCNN: Configured PyTorch Faster R-CNN model.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        min_size=min_size,
        max_size=max_size,
    )

    # Extract number of input features for the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace original head with custom FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    logger.info(
        f"Built Faster R-CNN (ResNet-50-FPN) with {num_classes} classes "
        f"(pretrained={pretrained}, trainable_layers={trainable_backbone_layers})"
    )
    return model
