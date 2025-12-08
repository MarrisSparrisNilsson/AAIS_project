# CPU-only Dockerfile for MLflow Fine-tuned Model with FastAPI
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY extract_info.py .

# Copy MLflow database and model artifacts
# Note: mlruns/ directory should be copied or mounted as volume
COPY mlflow.db .
COPY mlruns/ ./mlruns/

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Run FastAPI server
CMD ["python", "app.py"]

