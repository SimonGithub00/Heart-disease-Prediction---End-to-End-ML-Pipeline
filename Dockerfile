FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data  # optional; or mount a volume in prod
ENV PYTHONUNBUFFERED=1

# Default: train (override with `docker run ... python src/monitor.py`, etc.)
CMD ["python", "src/train.py"]
