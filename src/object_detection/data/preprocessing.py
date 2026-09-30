"""Data preprocessing and dataset splitting module."""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


def prepare_and_split_dataset(
    valid_samples: List[Dict[str, Any]],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Splits dataset into train, val, and test partitions, copies images and saves split metadata.

    Args:
        valid_samples: List of validated sample metadata dicts.
        output_dir: Base processed data directory (`data/processed`).
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Testing split ratio.
        seed: Random seed for shuffling.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping split names ('train', 'val', 'test') to sample lists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    indices = np.arange(len(valid_samples))
    np.random.shuffle(indices)

    total = len(valid_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": [valid_samples[i] for i in indices[:train_end]],
        "val": [valid_samples[i] for i in indices[train_end:val_end]],
        "test": [valid_samples[i] for i in indices[val_end:]]
    }

    for split_name, samples in splits.items():
        split_dir = output_dir / split_name
        split_images_dir = split_dir / "images"
        split_images_dir.mkdir(parents=True, exist_ok=True)

        processed_samples = []
        for sample in samples:
            src_img = Path(sample["image_path"])
            dest_img = split_images_dir / src_img.name
            shutil.copy2(src_img, dest_img)

            processed_sample = {
                "image_path": str(dest_img),
                "width": sample["width"],
                "height": sample["height"],
                "objects": sample["objects"]
            }
            processed_samples.append(processed_sample)

        metadata_path = split_dir / "annotations.json"
        with open(metadata_path, "w") as f:
            json.dump(processed_samples, f, indent=2)

        logger.info(f"Saved {len(processed_samples)} samples to processed split: {split_name}")

    return splits


def visualize_sample_annotations(
    sample: Dict[str, Any],
    output_path: Path,
    class_colors: Dict[str, Tuple[int, int, int]] = None
) -> None:
    """Draws ground truth bounding boxes and class labels on a sample image for visual sanity checking.

    Args:
        sample: Sample metadata dictionary containing image_path and objects.
        output_path: Path to save the annotated visualization image.
        class_colors: Color mapping for classes.
    """
    if class_colors is None:
        class_colors = {
            "helmet": (255, 215, 0),
            "head": (255, 105, 180),
            "person": (30, 144, 255),
            "vest": (255, 69, 0)
        }

    img_path = Path(sample["image_path"])
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for obj in sample["objects"]:
        cls_name, bbox = obj[0], obj[1]
        xmin, ymin, xmax, ymax = bbox
        color = class_colors.get(cls_name, (0, 255, 0))

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        label_text = f"{cls_name}"
        draw.text((xmin + 4, ymin + 4), label_text, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Sample annotation visualization saved to {output_path}")
