"""Unit tests for dataset downloading, validation, and preprocessing."""
import json
from pathlib import Path
import pytest
from object_detection.config.configuration import ConfigManager
from object_detection.data.validation import DatasetValidator


def test_config_loading():
    config = ConfigManager("configs/config.yaml")
    assert config.project["name"] == "object-detection-fasterrcnn"
    assert config.dataset["num_classes"] == 4
    assert "helmet" in config.dataset["class_mapping"].values()


def test_raw_dataset_exists():
    raw_dir = Path("data/raw")
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    assert images_dir.exists()
    assert annotations_dir.exists()
    assert len(list(images_dir.glob("*.jpg"))) > 0


def test_dataset_validator():
    config = ConfigManager("configs/config.yaml")
    valid_classes = [name for id_, name in config.dataset["class_mapping"].items() if name != "__background__"]
    validator = DatasetValidator(raw_dir=Path("data/raw"), valid_classes=valid_classes)
    report = validator.validate_dataset()

    assert report["total_images"] > 0
    assert len(report["valid_samples"]) > 0
    assert report["total_annotations"] > 0
    assert len(report["corrupted_images"]) == 0
    assert len(report["invalid_boxes"]) == 0


def test_processed_splits_exist():
    processed_dir = Path("data/processed_real")
    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        assert split_dir.exists()
        json_path = split_dir / "annotations.json"
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)
            assert len(data) > 0
            assert "objects" in data[0]


def test_real_voc_report_and_held_out_split():
    report = json.loads(Path("artifacts/metrics/dataset_report.json").read_text())
    split = json.loads(Path("data/processed_real/split_report.json").read_text())
    assert report["class_distribution"] == {"helmet": 19747, "head": 6677, "person": 615}
    assert split["test"] == 1766
    assert split["leakage_check"] == "passed"
