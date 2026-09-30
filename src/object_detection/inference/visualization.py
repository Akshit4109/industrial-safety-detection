"""Visualization utility for drawing predicted bounding boxes, labels, and confidence scores."""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


def draw_bounding_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
    class_mapping: Dict[int, str],
    confidence_threshold: float = 0.5,
    class_colors: Dict[str, Tuple[int, int, int]] = None
) -> Image.Image:
    """Draws predicted bounding boxes, labels, and confidence scores onto a PIL Image canvas.

    Args:
        image: Source PIL Image.
        boxes: List of predicted bounding box coordinates [[xmin, ymin, xmax, ymax], ...].
        labels: List of class IDs.
        scores: List of confidence scores (between 0.0 and 1.0).
        class_mapping: Dict mapping class IDs to class names.
        confidence_threshold: Minimum confidence score to render.
        class_colors: Color palette dict.

    Returns:
        Image.Image: Annotated PIL Image.
    """
    if class_colors is None:
        class_colors = {
            "helmet": (255, 215, 0),    # Gold/Yellow
            "head": (255, 105, 180),    # Pink
            "person": (30, 144, 255),   # Blue
            "vest": (255, 69, 0)        # Orange
        }

    annotated_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_img)

    for box, label_id, score in zip(boxes, labels, scores):
        if score < confidence_threshold:
            continue

        cls_name = class_mapping.get(label_id, f"Class_{label_id}")
        color = class_colors.get(cls_name, (0, 255, 0))
        xmin, ymin, xmax, ymax = box

        # Draw thick bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Render class name and score percentage
        caption = f"{cls_name} {score * 100:.1f}%"
        draw.rectangle([xmin, max(0, ymin - 18), xmin + len(caption) * 8, ymin], fill=color)
        draw.text((xmin + 4, max(0, ymin - 16)), caption, fill=(0, 0, 0))

    return annotated_img
