"""Production FastAPI Microservice for Faster R-CNN Object Detection."""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from api.schemas import PredictionResponse, HealthResponse, ModelInfoResponse
from api.services import InferenceService
from object_detection.inference.compliance import assess_compliance
from object_detection.utils.logger import setup_logger

logger = setup_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager initializing Faster R-CNN model weights at startup."""
    logger.info("=== Starting Faster R-CNN Object Detection FastAPI Service ===")
    service = InferenceService.get_instance()
    try:
        service.load_model()
        logger.info("Faster R-CNN model ready for HTTP requests.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
    yield
    logger.info("=== Shutting Down Faster R-CNN Object Detection Service ===")


app = FastAPI(
    title="Faster R-CNN Industrial Safety Object Detection API",
    description="Production-grade REST microservice performing real-time object detection on industrial images.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for browser integration & web client frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
    return response


@app.get("/", tags=["Root"])
async def root():
    return {
        "title": "Production-Grade Object Detection System",
        "description": "Faster R-CNN Object Detection API for Construction & Industrial Safety Monitoring",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    service = InferenceService.get_instance()
    return HealthResponse(
        status="ok" if service.model_loaded else "degraded",
        device=str(service.device),
        model_loaded=service.model_loaded,
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def model_info():
    service = InferenceService.get_instance()
    return ModelInfoResponse(
        model_loaded=service.model_loaded,
        architecture=service.config.get("model", {}).get("name", "fasterrcnn_resnet50_fpn"),
        classes=service.class_mapping,
        checkpoint_path=str(service.checkpoint_path),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_objects(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG)"),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Upload a valid image file (image/jpeg, image/png)."
        )

    try:
        image_bytes = await file.read()
        service = InferenceService.get_instance()
        result = service.predict(image_bytes, confidence_threshold=confidence_threshold)

        return PredictionResponse(
            success=result["success"],
            num_detections=result["num_detections"],
            predictions=result["predictions"],
            inference_time_ms=result["inference_time_ms"],
            timestamp=result["timestamp"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error during /predict request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Object detection inference failure: {str(e)}"
        )


@app.post(
    "/predict/visualize",
    responses={200: {"content": {"image/jpeg": {}}}},
    tags=["Inference"]
)
async def predict_visualize(
    file: UploadFile = File(..., description="Target image file"),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence score threshold")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image."
        )

    try:
        image_bytes = await file.read()
        service = InferenceService.get_instance()
        output_jpeg_bytes = service.predict_visualized(
            image_bytes,
            confidence_threshold=confidence_threshold
        )
        return Response(content=output_jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visualization error: {str(e)}"
        )


@app.post("/compliance", tags=["Inference"])
async def compliance(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File provided is not a valid image.")
    try:
        result = InferenceService.get_instance().predict(await file.read(), confidence_threshold)
        return {"detections": result["predictions"], "compliance": assess_compliance(result["predictions"]), "inference_time_ms": result["inference_time_ms"]}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
