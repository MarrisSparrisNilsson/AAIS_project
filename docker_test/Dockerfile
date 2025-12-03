FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git wget nano

# Copy your code
WORKDIR /app
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install papermill

# Expose API port
EXPOSE 8000

# Run FastAPI app
CMD ["python", "main.py"]