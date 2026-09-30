"""Pytest integration suite for FastAPI REST API endpoints."""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app
from api.services import InferenceService
from PIL import ImageDraw


def create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert data["docs_url"] == "/docs"


def test_health_check_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # No real checkpoint is fabricated before the first training run.
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert "device" in data


class FakeInferenceService:
    model_loaded = True
    device = "cpu"
    class_mapping = {0: "__background__", 1: "helmet", 2: "head", 3: "person"}
    checkpoint_path = "test.pth"
    config = {"model": {"name": "fasterrcnn_resnet50_fpn"}}

    def predict(self, image_bytes, confidence_threshold=0.5):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return {"success": True, "num_detections": 1, "predictions": [{"class_id": 1, "class_name": "helmet", "confidence": 0.9, "box": {"xmin": 1, "ymin": 1, "xmax": 30, "ymax": 30}}], "inference_time_ms": 1.0, "timestamp": "2026-01-01T00:00:00+00:00", "pil_image": image, "raw_boxes": [{"xmin": 1, "ymin": 1, "xmax": 30, "ymax": 30}], "raw_labels": [1], "raw_scores": [0.9]}

    def predict_visualized(self, image_bytes, confidence_threshold=0.5):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ImageDraw.Draw(image).rectangle([1, 1, 30, 30], outline="red")
        buf = io.BytesIO(); image.save(buf, format="JPEG"); return buf.getvalue()


@pytest.fixture
def fake_service(monkeypatch):
    service = FakeInferenceService()
    monkeypatch.setattr(InferenceService, "get_instance", classmethod(lambda cls, config_path="configs/config.yaml": service))
    return service


def test_predict_endpoint_success(fake_service):
    with TestClient(app) as client:
        img_bytes = create_test_image_bytes()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict?confidence_threshold=0.3", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "num_detections" in data
        assert "predictions" in data
        assert "inference_time_ms" in data


def test_predict_visualize_endpoint_success(fake_service):
    with TestClient(app) as client:
        img_bytes = create_test_image_bytes()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict/visualize?confidence_threshold=0.3", files=files)

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0


def test_predict_invalid_file_type():
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]


def test_model_info_endpoint(fake_service):
    with TestClient(app) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["classes"]["1"] == "helmet"
