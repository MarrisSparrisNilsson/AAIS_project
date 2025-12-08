# Docker Setup for Invoice Extraction API

## What This Does

This Docker setup packages your MLflow fine-tuned model and FastAPI server into a container that runs on **CPU**.

## Step-by-Step Process

### Step 1: Build Docker Image
```bash
docker build -t invoice-api:latest .
```

**What happens:**
- Downloads Python 3.13 base image (~200 MB)
- Installs system dependencies
- Installs Python packages from `requirements.txt` (~2-3 GB)
- Copies your code and MLflow model files
- Creates image (~8-12 GB total for CPU)

### Step 2: Run Container
```bash
docker run -p 8000:8000 invoice-api:latest
```

**What happens:**
- Starts container from image
- Loads MLflow model from `mlruns/` directory
- Starts FastAPI server on port 8000
- Model runs on CPU (slower than GPU but works)

### Step 3: Test API
```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict/path?image_path=/path/to/image.png"
```

## Using Docker Compose (Easier)

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

## Important Notes

### CPU vs GPU
- **CPU**: Works but slow (10-50x slower). No special setup needed.
- **GPU**: Faster but requires CUDA base image and `--gpus all` flag.

### Model Location
- Model must be in `mlruns/` directory
- `mlflow.db` must exist
- If model is missing, re-save it from notebook first

### File Sizes
- **CPU Image**: ~8-12 GB
- **GPU Image**: ~15-25 GB (if you upgrade later)

## Troubleshooting

**Model not loading:**
- Check `mlruns/` directory exists in container
- Verify model was saved with compatible Python version
- Check logs: `docker-compose logs`

**Port already in use:**
- Change port in `docker-compose.yml`: `"8001:8000"`

**Out of memory:**
- Model needs ~4-8 GB RAM on CPU
- Increase Docker memory limit

## Commands Summary

```bash
# Build image
docker build -t invoice-api:latest .

# Run container
docker run -p 8000:8000 invoice-api:latest

# Or use docker-compose
docker-compose up -d

# View API docs
open http://localhost:8000/docs
```

