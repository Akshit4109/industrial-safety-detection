"""Object Detection Metrics Engine calculating IoU, Precision, Recall, and mAP@50."""
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax].

    Args:
        boxA: Bounding box A coordinates [xmin, ymin, xmax, ymax].
        boxB: Bounding box B coordinates [xmin, ymin, xmax, ymax].

    Returns:
        float: Intersection over Union value between 0.0 and 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0

    return interArea / unionArea


class ObjectDetectionEvaluator:
    """Evaluates object detection model predictions against ground truth annotations."""

    def __init__(self, class_mapping: Dict[int, str], iou_threshold: float = 0.5, include_coco_metrics: bool = True) -> None:
        self.class_mapping = class_mapping
        self.id_to_class = {id_: name for id_, name in class_mapping.items() if name != "__background__"}
        self.iou_threshold = iou_threshold
        self.include_coco_metrics = include_coco_metrics

    def evaluate_predictions(
        self,
        all_targets: List[Dict[str, torch.Tensor]],
        all_predictions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Calculates mAP@50, Precision, Recall, and Per-Class Metrics across all evaluation samples.

        Args:
            all_targets: Ground truth target dictionaries.
            all_predictions: Model prediction dictionaries.

        Returns:
            Dict[str, Any]: Complete evaluation metrics report dictionary.
        """
        per_class_stats = {
            cls_name: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "ious": []}
            for cls_name in self.id_to_class.values()
        }

        for target, pred in zip(all_targets, all_predictions):
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()

            pred_boxes = pred["boxes"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()

            for cls_id, cls_name in self.id_to_class.items():
                gt_indices = np.where(gt_labels == cls_id)[0]
                pred_indices = np.where(pred_labels == cls_id)[0]

                cls_gt_boxes = gt_boxes[gt_indices]
                cls_pred_boxes = pred_boxes[pred_indices]
                cls_pred_scores = pred_scores[pred_indices]

                per_class_stats[cls_name]["gt_count"] += len(cls_gt_boxes)

                if len(cls_pred_boxes) == 0:
                    per_class_stats[cls_name]["fn"] += len(cls_gt_boxes)
                    continue

                # Sort predictions by confidence score descending
                sorted_order = np.argsort(-cls_pred_scores)
                cls_pred_boxes = cls_pred_boxes[sorted_order]

                gt_matched = set()
                for p_box in cls_pred_boxes:
                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx, g_box in enumerate(cls_gt_boxes):
                        if gt_idx in gt_matched:
                            continue
                        iou = calculate_iou(p_box.tolist(), g_box.tolist())
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= self.iou_threshold and best_gt_idx != -1:
                        per_class_stats[cls_name]["tp"] += 1
                        per_class_stats[cls_name]["ious"].append(best_iou)
                        gt_matched.add(best_gt_idx)
                    else:
                        per_class_stats[cls_name]["fp"] += 1

                unmatched_gt = len(cls_gt_boxes) - len(gt_matched)
                per_class_stats[cls_name]["fn"] += max(0, unmatched_gt)

        # Compute overall and per-class summary metrics
        per_class_metrics = {}
        active_aps = []
        total_tp, total_fp, total_fn = 0, 0, 0

        for cls_name, stats in per_class_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            mean_iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
            ap = precision * recall

            per_class_metrics[cls_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "ap50": round(ap, 4),
                "mean_iou": round(mean_iou, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "ground_truth_count": stats["gt_count"]
            }

            if stats["gt_count"] > 0:
                active_aps.append(ap)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        mAP_50 = float(np.mean(active_aps)) if active_aps else 0.0

        report = {
            "mAP_50": round(mAP_50, 4),
            "overall_precision": round(overall_precision, 4),
            "overall_recall": round(overall_recall, 4),
            "iou_threshold": self.iou_threshold,
            "per_class_metrics": per_class_metrics
        }

        if self.include_coco_metrics:
            threshold_reports = []
            for threshold in np.arange(0.5, 1.0, 0.05):
                evaluator = ObjectDetectionEvaluator(self.class_mapping, float(threshold), include_coco_metrics=False)
                threshold_reports.append(evaluator.evaluate_predictions(all_targets, all_predictions))
            report["mAP_50_95"] = round(float(np.mean([item["mAP_50"] for item in threshold_reports])), 4)
            for class_name, values in per_class_metrics.items():
                values["ap50_95"] = round(float(np.mean([item["per_class_metrics"][class_name]["ap50"] for item in threshold_reports])), 4)

        logger.info(f"=== Model Evaluation Summary (IoU threshold={self.iou_threshold}) ===")
        logger.info(f"mAP@50: {report['mAP_50']:.4f}")
        logger.info(f"Overall Precision: {report['overall_precision']:.4f}")
        logger.info(f"Overall Recall: {report['overall_recall']:.4f}")
        return report
