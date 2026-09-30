"""Pydantic data validation schemas for FastAPI object detection REST API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates schema [xmin, ymin, xmax, ymax]."""
    xmin: float = Field(..., description="Top-left X coordinate")
    ymin: float = Field(..., description="Top-left Y coordinate")
    xmax: float = Field(..., description="Bottom-right X coordinate")
    ymax: float = Field(..., description="Bottom-right Y coordinate")


class DetectionItem(BaseModel):
    """Single object detection item response schema."""
    class_id: int = Field(..., description="Numerical class ID")
    class_name: str = Field(..., description="Human-readable object class name")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    box: BoundingBox = Field(..., description="Bounding box coordinates")


class PredictionResponse(BaseModel):
    """Object detection API response model."""
    success: bool = Field(..., description="Status flag indicating prediction success")
    num_detections: int = Field(..., description="Number of detected objects above threshold")
    predictions: List[DetectionItem] = Field(..., description="List of detected object items")
    inference_time_ms: float = Field(..., description="Inference execution latency in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class HealthResponse(BaseModel):
    """Microservice health check schema."""
    status: str = Field(..., description="System status string")
    device: str = Field(..., description="Active PyTorch execution device")
    model_loaded: bool = Field(..., description="Model weight loading status flag")
    version: str = Field("1.0.0", description="API version")


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    architecture: str
    classes: dict
    checkpoint_path: str
