"""Unit tests for calculate_iou, ObjectDetectionEvaluator, and prediction visualizer."""
import pytest
import torch
from PIL import Image
from object_detection.config.configuration import ConfigManager
from object_detection.evaluation.metrics import calculate_iou, ObjectDetectionEvaluator
from object_detection.inference.visualization import draw_bounding_boxes


def test_calculate_iou_exact():
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert calculate_iou(box1, box2) == 1.0


def test_calculate_iou_partial():
    box1 = [0, 0, 10, 10]  # area = 100
    box2 = [5, 0, 15, 10]  # area = 100, inter = 50, union = 150
    assert abs(calculate_iou(box1, box2) - 0.3333) < 0.001


def test_calculate_iou_no_overlap():
    box1 = [0, 0, 5, 5]
    box2 = [10, 10, 20, 20]
    assert calculate_iou(box1, box2) == 0.0


def test_evaluator_perfect_match():
    config = ConfigManager("configs/config.yaml")
    evaluator = ObjectDetectionEvaluator(config.dataset["class_mapping"], iou_threshold=0.5)

    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1])  # helmet
    }]
    predictions = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.95])
    }]

    report = evaluator.evaluate_predictions(targets, predictions)
    assert report["mAP_50"] == 1.0
    assert report["overall_precision"] == 1.0
    assert report["overall_recall"] == 1.0


def test_visualization_drawing():
    config = ConfigManager("configs/config.yaml")
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    boxes = [[10.0, 10.0, 50.0, 50.0]]
    labels = [1]
    scores = [0.92]

    annotated_img = draw_bounding_boxes(
        image=img,
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_mapping=config.dataset["class_mapping"],
        confidence_threshold=0.5
    )
    assert isinstance(annotated_img, Image.Image)
    assert annotated_img.size == (100, 100)
