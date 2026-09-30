"""Pascal VOC ingestion for the supplied real Hard Hat Workers dataset."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _xml_to_sample(xml_path: Path, image_path: Path, valid_classes: set[str]) -> tuple[dict | None, list[str]]:
    """Parse and validate one VOC XML/image pair; return errors instead of hiding them."""
    errors: list[str] = []
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        return None, [f"corrupted image {image_path}: {exc}"]
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return None, [f"malformed XML {xml_path}: {exc}"]

    objects = []
    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if name not in valid_classes:
            errors.append(f"unknown class '{name}' in {xml_path.name}")
            continue
        if box is None:
            errors.append(f"missing bndbox #{index} in {xml_path.name}")
            continue
        try:
            coords = [float(box.findtext(key, "")) for key in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError:
            errors.append(f"non-numeric bndbox #{index} in {xml_path.name}")
            continue
        xmin, ymin, xmax, ymax = coords
        # A small number of Roboflow VOC files use inclusive end coordinates one
        # pixel beyond PIL's exclusive image bounds. Clip those safely and record it.
        clipped = [max(0.0, min(xmin, float(width))), max(0.0, min(ymin, float(height))),
                   max(0.0, min(xmax, float(width))), max(0.0, min(ymax, float(height)))]
        if clipped != coords:
            errors.append(f"clipped bndbox {coords} to {clipped} in {xml_path.name}")
        xmin, ymin, xmax, ymax = clipped
        if not (xmin < xmax and ymin < ymax):
            errors.append(f"invalid bndbox {coords} in {xml_path.name} ({width}x{height})")
            continue
        objects.append([name, coords])
    if not objects:
        errors.append(f"empty valid annotation in {xml_path.name}")
        return None, errors
    return {
        "image_path": str(image_path), "xml_path": str(xml_path), "width": width, "height": height,
        "objects": objects,
        "difficult_count": sum(obj.findtext("difficult", "0") == "1" for obj in root.findall("object")),
        "occluded_count": sum(obj.findtext("occluded", "0") == "1" for obj in root.findall("object")),
    }, errors


def inspect_voc_dataset(root: Path, valid_classes: Iterable[str]) -> Dict[str, Any]:
    """Validate a VOC root with train/test folders and return a serializable report/samples."""
    root = Path(root)
    valid = set(valid_classes)
    report: Dict[str, Any] = {"dataset_root": str(root), "splits": {}, "errors": [], "class_distribution": Counter(), "dimensions": Counter(), "total_annotations": 0}
    for split in ("train", "val", "valid", "test"):
        directory = root / split
        if not directory.exists():
            continue
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        samples = []
        missing_xml = []
        for image in images:
            xml = image.with_suffix(".xml")
            if not xml.exists():
                missing_xml.append(str(image))
                continue
            sample, errors = _xml_to_sample(xml, image, valid)
            report["errors"].extend(errors)
            if sample:
                samples.append(sample)
                report["dimensions"][f"{sample['width']}x{sample['height']}"] += 1
                for name, _ in sample["objects"]:
                    report["class_distribution"][name] += 1
                    report["total_annotations"] += 1
        report["splits"][split] = {"image_count": len(images), "xml_count": len(list(directory.glob("*.xml"))), "valid_samples": samples, "missing_xml": missing_xml}
    report["class_distribution"] = dict(report["class_distribution"])
    report["image_dimensions"] = dict(report.pop("dimensions"))
    report["total_images"] = sum(info["image_count"] for info in report["splits"].values())
    return report


def save_dataset_report(report: Dict[str, Any], path: Path) -> None:
    """Persist report without the large internal sample lists."""
    compact = {**report, "splits": {name: {k: v for k, v in info.items() if k != "valid_samples"} | {"valid_image_count": len(info["valid_samples"])} for name, info in report["splits"].items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compact, indent=2), encoding="utf-8")
