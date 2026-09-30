"""Validate supplied VOC data and make reproducible train/validation/test manifests."""
import json
from pathlib import Path
import numpy as np

from object_detection.config.configuration import ConfigManager
from object_detection.data.voc import inspect_voc_dataset, save_dataset_report
from object_detection.utils.logger import setup_logger

logger = setup_logger("prepare_real_data")


def _write_manifest(samples, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "annotations.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")


def main() -> None:
    config = ConfigManager("configs/config.yaml")
    classes = [name for name in config.dataset["class_mapping"].values() if name != "__background__"]
    report = inspect_voc_dataset(Path(config.dataset["raw_dir"]), classes)
    save_dataset_report(report, Path("artifacts/metrics/dataset_report.json"))
    fatal_errors = [error for error in report["errors"] if not error.startswith("clipped bndbox")]
    if fatal_errors:
        raise ValueError(f"VOC validation found {len(fatal_errors)} fatal errors; see artifacts/metrics/dataset_report.json")
    if "train" not in report["splits"] or "test" not in report["splits"]:
        raise ValueError("Expected supplied VOC train and test folders")
    samples = list(report["splits"]["train"]["valid_samples"])
    rng = np.random.default_rng(config.project["seed"])
    rng.shuffle(samples)
    val_count = round(len(samples) * config.dataset["val_split"])
    val, train = samples[:val_count], samples[val_count:]
    test = report["splits"]["test"]["valid_samples"]
    names = [{Path(s["image_path"]).name for s in group} for group in (train, val, test)]
    if names[0] & names[1] or names[0] & names[2] or names[1] & names[2]:
        raise ValueError("Train/validation/test leakage detected")
    output = Path(config.dataset["processed_dir"])
    for split, split_samples in (("train", train), ("val", val), ("test", test)):
        _write_manifest(split_samples, output / split)
    (output / "split_report.json").write_text(json.dumps({"train": len(train), "val": len(val), "test": len(test), "seed": config.project["seed"], "validation_source": "random split of supplied train only", "leakage_check": "passed"}, indent=2), encoding="utf-8")
    logger.info("Real-data manifests written: train=%s val=%s test=%s", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
