"""Run a bounded real-data Faster R-CNN forward/backward test on Apple MPS."""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from object_detection.config.configuration import ConfigManager
from object_detection.data.dataset import ObjectDetectionDataset, collate_fn
from object_detection.models.model_factory import ModelFactory
from object_detection.utils.device import get_device


def _tensor_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _gradient_norm(parameters) -> torch.Tensor:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=2, choices=range(1, 4))
    args = parser.parse_args()
    device = get_device("mps")
    if device.type != "mps":
        raise RuntimeError("MPS is unavailable in this environment; no smoke test was run.")
    config = ConfigManager("configs/config.yaml")
    training = config.training
    accumulation_steps = max(1, int(training.get("gradient_accumulation_steps", 1)))
    max_gradient_norm = float(training.get("max_gradient_norm", 0.0))
    dataset = ObjectDetectionDataset(Path(config.dataset["processed_dir"]) / "train" / "annotations.json", config.dataset["class_mapping"], is_train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = ModelFactory.create_model(config.get_all(), device=device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=training["learning_rate"], momentum=training["momentum"], weight_decay=training["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    for index, (images, targets) in enumerate(loader, 1):
        start = time.perf_counter()
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        input_finite = all(_tensor_finite(image) for image in images)
        target_boxes = torch.cat([target["boxes"] for target in targets])
        labels = sorted({label for target in targets for label in target["labels"].tolist()})
        losses = model(images, targets)
        loss = sum(losses.values())
        component_values = {name: value.item() for name, value in losses.items()}
        losses_finite = all(_tensor_finite(value) for value in losses.values()) and _tensor_finite(loss)
        if not input_finite or not _tensor_finite(target_boxes) or not losses_finite:
            raise FloatingPointError(f"Non-finite input/target/loss at batch {index}")
        (loss / accumulation_steps).backward()
        gradients_finite = all(parameter.grad is None or _tensor_finite(parameter.grad) for parameter in parameters)
        gradient_norm = _gradient_norm(parameters)
        if not gradients_finite or not _tensor_finite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient at batch {index}")
        should_step = index % accumulation_steps == 0 or index == args.batches
        if should_step:
            if max_gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        print({"batch": index, "total_loss": loss.item(), "loss_components": component_values, "learning_rate": optimizer.param_groups[0]["lr"], "images": len(images), "targets": sum(len(target["labels"]) for target in targets), "labels": labels, "box_min": target_boxes.min().item(), "box_max": target_boxes.max().item(), "inputs_finite": input_finite, "losses_finite": losses_finite, "gradients_finite": gradients_finite, "pre_clip_gradient_norm": gradient_norm.item(), "optimizer_step": should_step, "time_s": round(time.perf_counter() - start, 2)})
        del losses, loss, images, targets
        torch.mps.empty_cache()
        if index >= args.batches:
            break


if __name__ == "__main__":
    main()
