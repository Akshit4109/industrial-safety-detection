"""CLI script to execute model evaluation."""
from pathlib import Path

from object_detection.config.configuration import ConfigManager
from object_detection.evaluation.evaluator import run_evaluation
from object_detection.utils.device import get_device
from object_detection.utils.logger import setup_logger

logger = setup_logger("evaluate_script")


def main():
    logger.info("=== Initializing Object Detection Model Evaluation ===")
    config = ConfigManager("configs/config.yaml")
    device = get_device()

    checkpoint_path = Path(config.training["save_dir"]) / "best_model.pth"
    output_metrics_path = Path("artifacts/metrics/evaluation.json")
    output_predictions_dir = Path("artifacts/predictions")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint model file not found at {checkpoint_path}! Run scripts/train.py first.")

    report = run_evaluation(
        config=config.get_all(),
        checkpoint_path=checkpoint_path,
        output_metrics_path=output_metrics_path,
        output_predictions_dir=output_predictions_dir,
        device=device
    )

    logger.info("=== Final Evaluation Metrics ===")
    logger.info(f"mAP@50: {report['mAP_50']}")
    logger.info(f"Precision: {report['overall_precision']}")
    logger.info(f"Recall: {report['overall_recall']}")


if __name__ == "__main__":
    main()
