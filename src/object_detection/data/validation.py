"""Dataset validation and summary statistics module."""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image, ImagePath
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates object detection dataset annotations, image integrity, and bounding box formats."""

    def __init__(self, raw_dir: Path, valid_classes: List[str]) -> None:
        self.raw_dir = Path(raw_dir)
        self.images_dir = self.raw_dir / "images"
        self.annotations_dir = self.raw_dir / "annotations"
        self.valid_classes = set(valid_classes)

    def validate_dataset(self) -> Dict[str, Any]:
        """Runs validation checks across images and Pascal VOC XML annotations.

        Returns:
            Dict[str, Any]: Validation report detailing statistics and corruption issues.
        """
        report = {
            "total_images": 0,
            "total_annotations": 0,
            "valid_samples": [],
            "corrupted_images": [],
            "missing_annotations": [],
            "invalid_boxes": [],
            "unknown_classes": [],
            "class_counts": {cls_name: 0 for cls_name in self.valid_classes},
            "box_areas": [],
        }

        if not self.images_dir.exists() or not self.annotations_dir.exists():
            logger.error(f"Images or Annotations directory missing in {self.raw_dir}")
            return report

        image_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        report["total_images"] = len(image_files)

        for img_path in image_files:
            stem = img_path.stem
            xml_path = self.annotations_dir / f"{stem}.xml"

            # Check missing annotation XML
            if not xml_path.exists():
                report["missing_annotations"].append(str(img_path))
                continue

            # Check image integrity
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Corrupted image detected {img_path}: {e}")
                report["corrupted_images"].append(str(img_path))
                continue

            # Parse XML annotation
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                logger.warning(f"Invalid XML annotation file {xml_path}: {e}")
                report["corrupted_images"].append(str(xml_path))
                continue

            sample_valid = True
            objects_in_sample = []

            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in self.valid_classes:
                    report["unknown_classes"].append((stem, cls_name))
                    sample_valid = False
                    continue

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # Validate bounding box coordinates
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmin >= xmax or ymin >= ymax:
                    report["invalid_boxes"].append((stem, [xmin, ymin, xmax, ymax]))
                    sample_valid = False
                    continue

                area = (xmax - xmin) * (ymax - ymin)
                report["box_areas"].append(area)
                report["class_counts"][cls_name] += 1
                report["total_annotations"] += 1
                objects_in_sample.append((cls_name, [xmin, ymin, xmax, ymax]))

            if sample_valid and len(objects_in_sample) > 0:
                report["valid_samples"].append({
                    "image_path": str(img_path),
                    "xml_path": str(xml_path),
                    "width": width,
                    "height": height,
                    "objects": objects_in_sample
                })

        logger.info("=== Dataset Validation Complete ===")
        logger.info(f"Total Images: {report['total_images']}")
        logger.info(f"Valid Samples: {len(report['valid_samples'])}")
        logger.info(f"Total Bounding Box Annotations: {report['total_annotations']}")
        logger.info(f"Class Distribution: {report['class_counts']}")
        if report["box_areas"]:
            logger.info(f"Mean BBox Area: {np.mean(report['box_areas']):.2f} px^2")

        return report
