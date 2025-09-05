# Multi-stage for smaller images + reproducibility

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (build-essentials optional; remove if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Separate layer for deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Source + (optional) data mount at runtime
COPY src ./src

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command can be overridden:
# e.g. docker run ... python src/evaluate.py
CMD ["python", "src/train.py"]
