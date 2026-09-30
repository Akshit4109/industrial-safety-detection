"""Unit tests for ObjectDetectionDataset, ModelFactory, and Faster R-CNN forward pass."""
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
from object_detection.config.configuration import ConfigManager
from object_detection.data.dataset import ObjectDetectionDataset, collate_fn
from object_detection.models.model_factory import ModelFactory
from object_detection.utils.device import get_device
import object_detection.utils.device as device_utils
from object_detection.training.trainer import Trainer


@pytest.fixture
def config():
    return ConfigManager("configs/config.yaml")


def test_dataset_loader(config):
    train_json = Path(config.dataset["processed_dir"]) / "train" / "annotations.json"
    dataset = ObjectDetectionDataset(
        annotations_json=train_json,
        class_mapping=config.dataset["class_mapping"],
        is_train=True
    )

    assert len(dataset) > 0
    img_tensor, target = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dim() == 3
    assert img_tensor.shape[0] == 3  # RGB channels
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape[1] == 4


def test_dataloader_collate(config):
    train_json = Path(config.dataset["processed_dir"]) / "train" / "annotations.json"
    dataset = ObjectDetectionDataset(
        annotations_json=train_json,
        class_mapping=config.dataset["class_mapping"],
        is_train=False
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    images, targets = next(iter(dataloader))
    assert len(images) == 2
    assert len(targets) == 2


def test_model_training_forward_pass(config):
    device = get_device()
    model = ModelFactory.create_model(config.get_all(), device=device)
    model.train()

    train_json = Path(config.dataset["processed_dir"]) / "train" / "annotations.json"
    dataset = ObjectDetectionDataset(
        annotations_json=train_json,
        class_mapping=config.dataset["class_mapping"],
        is_train=True
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    images, targets = next(iter(dataloader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)

    assert isinstance(loss_dict, dict)
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict
    assert "loss_objectness" in loss_dict
    assert "loss_rpn_box_reg" in loss_dict

    total_loss = sum(loss for loss in loss_dict.values())
    assert torch.isfinite(total_loss)


def test_model_eval_forward_pass(config):
    device = get_device()
    model = ModelFactory.create_model(config.get_all(), device=device)
    model.eval()

    train_json = Path(config.dataset["processed_dir"]) / "val" / "annotations.json"
    dataset = ObjectDetectionDataset(
        annotations_json=train_json,
        class_mapping=config.dataset["class_mapping"],
        is_train=False
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    images, _ = next(iter(dataloader))
    images = [img.to(device) for img in images]

    with torch.no_grad():
        predictions = model(images)

    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert "boxes" in predictions[0]
    assert "labels" in predictions[0]
    assert "scores" in predictions[0]


def test_auto_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device_utils.get_device("auto").type == "mps"


def test_gradient_accumulation_step_boundary():
    trainer = Trainer.__new__(Trainer)
    trainer.accumulation_steps = 2
    assert trainer.should_optimizer_step(0, 2) is False
    assert trainer.should_optimizer_step(1, 2) is True
