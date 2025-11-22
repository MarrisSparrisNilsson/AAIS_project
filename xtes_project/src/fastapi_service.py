

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wine Quality API", version="1.0.0")

class WineFeatures(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    status: str

model = None

def load_model():
    global model
    try:
        # Use 'mlflow' as hostname (Docker service name) instead of 'localhost'
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"🔗 Connecting to MLFlow at: {mlflow_tracking_uri}")
        model_uri = "models:/WineQualityModel/production"
        print(f".........Trying to load model from: {model_uri}")

        # Try to load production model, fallback to latest
        try:
            model_uri = "mlruns/1/models/m-f5b80dead525445a893f34fd139970e9/artifacts/model.pkl"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Loaded production model from MLFlow")
        except Exception as e:
            logger.warning(f"⚠️ Production model not found, trying latest: {e}")
            model_uri = "models:/WineQualityModel/latest" 
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Loaded latest model from MLFlow")
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Wine Quality Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if len(features.features) != 11:
            raise HTTPException(status_code=400, detail="Expected 11 features")
        
        feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        input_df = pd.DataFrame([features.features], columns=feature_names)
        prediction = model.predict(input_df)
        
        logger.info(f"✅ Prediction made: {prediction[0]:.2f}")
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
