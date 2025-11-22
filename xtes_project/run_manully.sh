#!/bin/bash

echo "🍷 Starting Wine Quality MLOps System..."
echo "=========================================="

echo ""
echo "1️⃣  Generating realistic wine data..."
python src/generate_wine_data.py 

echo ""
echo "2️⃣  Starting MLFlow (Terminal 1)..."
echo "   Run this in Terminal 1:"
echo "   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001"
echo ""
echo "   Then press Enter to continue..."
read

echo ""
echo "3️⃣  Training model (Terminal 2)..."
echo "   Run this in Terminal 2:"
echo "   python src/train_model.py"
echo ""
echo "   Then press Enter to continue..."
read

echo ""
echo "4️⃣  Starting API (Terminal 3)..."
echo "   Run this in Terminal 3:"
echo "   uvicorn src.fastapi_service:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "   Wait for 'Application startup complete', then press Enter..."
read

echo ""
echo "5️⃣  Testing API (Terminal 4)..."
echo "   Run this in Terminal 4 to test:"
echo "   curl -X POST \"http://localhost:8000/predict\" \\"
echo "        -H \"Content-Type: application/json\" \\"
echo "        -d '{\"features\": [7.4, 0.7, 0.1, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}'"
echo ""
echo "🎉 Done! Your MLOps system is running!"
echo "   MLFlow: http://localhost:5001"
echo "   API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"


##################################

docker-compose up --build -d # Built and started both containers in background
python src/generate_wine_data.py
python src/train_model.py





# Stop any running Docker containers
docker-compose down

# Stop any local services that might be running
pkill -f "mlflow server"
pkill -f "uvicorn"

# Clean up (optional - removes old data)
sudo rm -rf mlruns/ mlflow.db 2>/dev/null
mkdir -p mlruns            

# Find what's using port 8000
sudo lsof -i :8000

# Stop it
pkill -f "uvicorn src.fastapi_service"



# Start MLFlow terminal 1 
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Start FastAPI with auto-reload (developent mode) terminal 2
uvicorn src.fastapi_service:app --host 0.0.0.0 --port 8000 --reload

# Generate data terminal 3
python src/generate_data.py

# Train model
python src/train_model.py


curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [7.4, 0.7, 0.1, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}'










