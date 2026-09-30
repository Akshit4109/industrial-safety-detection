"""Custom PyTorch Dataset and DataLoader Collate utility for Object Detection."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ObjectDetectionDataset(Dataset):
    """PyTorch Dataset for Object Detection with bounding box transforms."""

    def __init__(
        self,
        annotations_json: Path,
        class_mapping: Dict[int, str],
        transforms: Optional[Callable] = None,
        is_train: bool = True
    ) -> None:
        """
        Args:
            annotations_json: Path to split annotations JSON.
            class_mapping: Mapping from class ID (int) to class name (str).
            transforms: Optional albumentations/torchvision transform pipeline.
            is_train: Whether dataset is used for training.
        """
        self.annotations_json = Path(annotations_json)
        if not self.annotations_json.exists():
            raise FileNotFoundError(f"Annotations JSON file not found at {self.annotations_json}")

        with open(self.annotations_json, "r") as f:
            self.samples = json.load(f)

        self.class_name_to_id = {name: id_ for id_, name in class_mapping.items()}
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        img_path = Path(sample["image_path"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        width, height = sample["width"], sample["height"]

        boxes = []
        labels = []
        areas = []

        for obj in sample["objects"]:
            cls_name, bbox = obj[0], obj[1]
            if cls_name in self.class_name_to_id:
                label_id = self.class_name_to_id[cls_name]
                xmin, ymin, xmax, ymax = bbox
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_id)
                areas.append((xmax - xmin) * (ymax - ymin))

        # Convert image to tensor [3, H, W] normalized to [0.0, 1.0]
        image_tensor = F.to_tensor(image)

        # Handle horizontal flip data augmentation for training
        if self.is_train and np.random.rand() > 0.5:
            image_tensor = F.hflip(image_tensor)
            flipped_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                new_xmin = width - xmax
                new_xmax = width - xmin
                flipped_boxes.append([new_xmin, ymin, new_xmax, ymax])
            boxes = flipped_boxes

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
        image_id_tensor = torch.tensor([idx], dtype=torch.int64)
        iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id_tensor,
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor
        }

        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Custom collate function to handle variable number of objects per image in batch.

    Args:
        batch: List of tuples (image_tensor, target_dict).

    Returns:
        Tuple of (list of image tensors, list of target dicts).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
