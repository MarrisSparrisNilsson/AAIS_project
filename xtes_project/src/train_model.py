# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

def setup_mlflow():
    """Setup MLFlow tracking"""
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Wine Quality Prediction****")

def load_data():
    """Load and prepare data"""
    df = pd.read_csv('data/wine_quality.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model():
    """Train model and log to MLFlow"""
    setup_mlflow()
    
    X_train, X_test, y_train, y_test = load_data()
    
    with mlflow.start_run():
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 38
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        })
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="WineQualityModel"
        )
        
        print("✅ Model trained and logged to MLFlow!")
        print(f"📈 Metrics - MSE: {mse:.3f}, R2: {r2:.3f}")
        
        return model

if __name__ == "__main__":
    train_and_log_model()