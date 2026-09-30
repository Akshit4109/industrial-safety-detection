"""Inference Service encapsulating Faster R-CNN model loading and image processing."""
import io
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torchvision.transforms.functional as F
from PIL import Image

from object_detection.config.configuration import ConfigManager
from object_detection.models.model_factory import ModelFactory
from object_detection.training.checkpoint import CheckpointManager
from object_detection.inference.visualization import draw_bounding_boxes
from object_detection.utils.device import get_device
from object_detection.utils.logger import setup_logger

logger = setup_logger("inference_service")


class InferenceService:
    """Singleton service for object detection API inference."""

    _instance: Optional["InferenceService"] = None

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_all()
        self.device = get_device()
        self.class_mapping = self.config.get("dataset", {}).get("class_mapping", {})

        self.model = None
        self.model_loaded = False
        self.checkpoint_path = Path(self.config.get("training", {}).get("save_dir", "artifacts/models")) / "best_model.pth"

    @classmethod
    def get_instance(cls, config_path: str = "configs/config.yaml") -> "InferenceService":
        if cls._instance is None:
            cls._instance = InferenceService(config_path)
        return cls._instance

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        if self.model_loaded and self.model is not None:
            return

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Model checkpoint file not found at {ckpt_file}!")

        logger.info(f"Loading object detection model from {ckpt_file} onto device {self.device}...")
        self.model = ModelFactory.create_model(self.config, device=self.device)
        checkpoint_mgr = CheckpointManager(ckpt_file.parent)
        checkpoint_mgr.load_checkpoint(self.model, checkpoint_path=ckpt_file, device=self.device)
        self.model.eval()
        self.model_loaded = True
        logger.info("Object detection model loaded and ready for API inference requests.")

    def predict(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Processes binary image bytes and returns structured prediction results.

        Args:
            image_bytes: Raw binary uploaded image bytes.
            confidence_threshold: Confidence score cutoff threshold.

        Returns:
            Dict[str, Any]: Structured prediction dictionary.
        """
        if not self.model_loaded or self.model is None:
            self.load_model()

        start_time = time.perf_counter()

        # Decode binary image bytes
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {e}")

        # Convert image to FloatTensor [0.0, 1.0]
        img_tensor = F.to_tensor(pil_img).to(self.device)

        with torch.no_grad():
            preds = self.model([img_tensor])[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        filtered_predictions = []
        for box, label_id, score in zip(boxes, labels, scores):
            if score < confidence_threshold:
                continue

            cls_name = self.class_mapping.get(int(label_id), f"Class_{int(label_id)}")
            filtered_predictions.append({
                "class_id": int(label_id),
                "class_name": cls_name,
                "confidence": round(float(score), 4),
                "box": {
                    "xmin": round(float(box[0]), 2),
                    "ymin": round(float(box[1]), 2),
                    "xmax": round(float(box[2]), 2),
                    "ymax": round(float(box[3]), 2)
                }
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "success": True,
            "num_detections": len(filtered_predictions),
            "predictions": filtered_predictions,
            "inference_time_ms": round(latency_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pil_image": pil_img,
            "raw_boxes": [p["box"] for p in filtered_predictions],
            "raw_labels": [p["class_id"] for p in filtered_predictions],
            "raw_scores": [p["confidence"] for p in filtered_predictions]
        }

    def predict_visualized(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5
    ) -> bytes:
        """Runs object detection and returns annotated JPEG image bytes.

        Args:
            image_bytes: Raw input image bytes.
            confidence_threshold: Minimum score threshold.

        Returns:
            bytes: Binary JPEG image bytes overlaid with bounding box predictions.
        """
        result = self.predict(image_bytes, confidence_threshold=confidence_threshold)
        pil_img = result["pil_image"]

        boxes = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in result["raw_boxes"]]
        labels = result["raw_labels"]
        scores = result["raw_scores"]

        annotated_img = draw_bounding_boxes(
            image=pil_img,
            boxes=boxes,
            labels=labels,
            scores=scores,
            class_mapping=self.class_mapping,
            confidence_threshold=confidence_threshold
        )

        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
