"""Evaluator Engine orchestrating model testing and metrics generation."""
import json
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from PIL import Image

from object_detection.data.dataset import ObjectDetectionDataset, collate_fn
from object_detection.models.model_factory import ModelFactory
from object_detection.training.checkpoint import CheckpointManager
from object_detection.evaluation.metrics import ObjectDetectionEvaluator
from object_detection.inference.visualization import draw_bounding_boxes
from object_detection.utils.logger import setup_logger

logger = setup_logger("evaluator")


def run_evaluation(
    config: Dict[str, Any],
    checkpoint_path: Path,
    output_metrics_path: Path,
    output_predictions_dir: Path,
    device: torch.device
) -> Dict[str, Any]:
    """Runs model evaluation on test dataset split and generates evaluation report and visualizations.

    Args:
        config: Central configuration dictionary.
        checkpoint_path: Path to best_model.pth checkpoint file.
        output_metrics_path: Path to save evaluation JSON report.
        output_predictions_dir: Directory to save visual prediction overlays.
        device: Active PyTorch hardware device.

    Returns:
        Dict[str, Any]: Calculated metric values dictionary.
    """
    dataset_cfg = config.get("dataset", {})
    class_mapping = dataset_cfg.get("class_mapping", {})
    processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed"))
    test_json = processed_dir / "test" / "annotations.json"

    logger.info(f"Loading test split dataset from {test_json}...")
    test_dataset = ObjectDetectionDataset(test_json, class_mapping, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Loading model checkpoint from {checkpoint_path}...")
    model = ModelFactory.create_model(config, device=device)
    checkpoint_mgr = CheckpointManager(checkpoint_path.parent)
    checkpoint_mgr.load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    all_targets = []
    all_predictions = []

    logger.info("Executing evaluation forward pass across test samples...")
    with torch.no_grad():
        for images, targets in test_loader:
            images_device = [img.to(device) for img in images]
            preds = model(images_device)

            all_targets.extend(targets)
            all_predictions.extend(preds)

    evaluator = ObjectDetectionEvaluator(class_mapping=class_mapping, iou_threshold=0.5)
    metrics_report = evaluator.evaluate_predictions(all_targets, all_predictions)

    # Save metrics JSON report
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Evaluation report saved to {output_metrics_path}")

    # Generate sample prediction visualization overlays
    output_predictions_dir.mkdir(parents=True, exist_ok=True)
    with open(test_json, "r") as f:
        test_samples = json.load(f)

    for i in range(min(5, len(test_samples))):
        sample = test_samples[i]
        img_path = Path(sample["image_path"])
        pred = all_predictions[i]

        boxes = pred["boxes"].cpu().numpy().tolist()
        labels = pred["labels"].cpu().numpy().tolist()
        scores = pred["scores"].cpu().numpy().tolist()

        img = Image.open(img_path)
        annotated_img = draw_bounding_boxes(
            image=img,
            boxes=boxes,
            labels=labels,
            scores=scores,
            class_mapping=class_mapping,
            confidence_threshold=0.3
        )
        save_path = output_predictions_dir / f"test_prediction_{i:02d}.jpg"
        annotated_img.save(save_path)

    logger.info(f"Sample prediction images saved to {output_predictions_dir}")
    return metrics_report
