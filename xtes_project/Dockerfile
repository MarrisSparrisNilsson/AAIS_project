FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "src.fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]