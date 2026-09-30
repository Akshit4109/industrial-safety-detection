# Production Dockerfile for Faster R-CNN Object Detection Microservice
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (OpenCV GL dependencies & curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and project configuration files
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY api/ /app/api/
COPY scripts/ /app/scripts/
COPY artifacts/ /app/artifacts/

# Install editable package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Healthcheck monitoring
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
