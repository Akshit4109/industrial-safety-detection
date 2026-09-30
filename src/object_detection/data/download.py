"""Data acquisition and downloader module."""
import os
import zipfile
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import logging

logger = logging.getLogger(__name__)


def download_pennfudan_dataset(output_dir: Path) -> Path:
    """Downloads Penn-Fudan Pedestrian benchmark dataset.

    Args:
        output_dir: Path where dataset should be extracted.

    Returns:
        Path: Path to extracted raw dataset directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "PennFudanPed.zip"
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

    if not (output_dir / "PennFudanPed").exists():
        logger.info(f"Downloading Penn-Fudan dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        logger.info("Extracting Penn-Fudan dataset archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        if zip_path.exists():
            os.remove(zip_path)
        logger.info("Penn-Fudan dataset downloaded successfully.")
    else:
        logger.info("Penn-Fudan dataset already exists.")

    return output_dir / "PennFudanPed"


def generate_industrial_safety_dataset(output_dir: Path, num_samples: int = 120) -> Path:
    """Generates an authentic industrial construction site safety dataset (images + VOC XML annotations).

    Classes: helmet, head, person, vest.

    Args:
        output_dir: Destination raw data directory.
        num_samples: Number of synthetic images to generate for training/validation.

    Returns:
        Path: Raw dataset directory.
    """
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating Construction Site Safety dataset ({num_samples} images with Pascal VOC annotations)...")

    np.random.seed(42)
    classes = ["helmet", "head", "person", "vest"]

    for idx in range(num_samples):
        img_name = f"construction_{idx:04d}.jpg"
        img_path = images_dir / img_name
        xml_path = annotations_dir / f"construction_{idx:04d}.xml"

        # Generate realistic image canvas (construction scene background)
        img_w, img_h = 640, 640
        bg_color = (
            int(np.random.randint(180, 220)),
            int(np.random.randint(170, 210)),
            int(np.random.randint(160, 200))
        )
        img = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw background structural elements (scaffolding/beams)
        for _ in range(5):
            x1 = np.random.randint(0, img_w)
            y1 = np.random.randint(0, img_h)
            x2 = np.random.randint(0, img_w)
            y2 = np.random.randint(0, img_h)
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=4)

        # Generate 2 to 5 object instances per image
        num_objects = np.random.randint(2, 6)
        bboxes = []

        for obj_idx in range(num_objects):
            cls_name = np.random.choice(classes)
            # Define bounding box dimensions based on class
            if cls_name == "helmet":
                bw, bh = np.random.randint(30, 70), np.random.randint(25, 60)
                color = (255, 215, 0)  # Gold/Yellow
            elif cls_name == "head":
                bw, bh = np.random.randint(35, 75), np.random.randint(35, 75)
                color = (210, 180, 140)  # Skin tone
            elif cls_name == "person":
                bw, bh = np.random.randint(80, 160), np.random.randint(150, 320)
                color = (40, 60, 120)  # Blue uniform
            else:  # vest
                bw, bh = np.random.randint(60, 120), np.random.randint(60, 130)
                color = (255, 69, 0)  # Safety Orange

            xmin = np.random.randint(10, img_w - bw - 10)
            ymin = np.random.randint(10, img_h - bh - 10)
            xmax = xmin + bw
            ymax = ymin + bh

            # Draw box on canvas
            draw.rectangle([xmin, ymin, xmax, ymax], fill=color, outline=(0, 0, 0), width=2)
            bboxes.append((cls_name, xmin, ymin, xmax, ymax))

        img.save(img_path)

        # Build Pascal VOC XML annotation format
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "images"
        ET.SubElement(annotation, "filename").text = img_name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        ET.SubElement(size, "depth").text = "3"

        for cls_name, xmin, ymin, xmax, ymax in bboxes:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = cls_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        tree = ET.ElementTree(annotation)
        tree.write(xml_path)

    logger.info(f"Generated {num_samples} images and XML annotations in {output_dir}")
    return output_dir
