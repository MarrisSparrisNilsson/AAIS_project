# src/generate_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

def generate_wine_data():
    """Generate realistic wine quality dataset"""
    np.random.seed(42)
    
    # Realistic wine feature ranges (based on UCI Wine Quality dataset)
    n_samples = 1000
    
    data = {
        'fixed_acidity': np.random.uniform(4.0, 15.0, n_samples),
        'volatile_acidity': np.random.uniform(0.1, 1.5, n_samples),
        'citric_acid': np.random.uniform(0.0, 1.0, n_samples),
        'residual_sugar': np.random.uniform(0.5, 15.0, n_samples),
        'chlorides': np.random.uniform(0.01, 0.2, n_samples),
        'free_sulfur_dioxide': np.random.uniform(1, 70, n_samples),
        'total_sulfur_dioxide': np.random.uniform(5, 250, n_samples),
        'density': np.random.uniform(0.98, 1.02, n_samples),
        'pH': np.random.uniform(2.8, 4.0, n_samples),
        'sulphates': np.random.uniform(0.3, 2.0, n_samples),
        'alcohol': np.random.uniform(8.0, 14.0, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic quality scores (3-8 range)
    # Based on real correlations from wine dataset
    quality = (
        0.5 * df['fixed_acidity'] +
        -1.5 * df['volatile_acidity'] +
        0.5 * df['citric_acid'] +
        0.1 * df['residual_sugar'] +
        -5.0 * df['chlorides'] +
        0.01 * df['free_sulfur_dioxide'] +
        0.002 * df['total_sulfur_dioxide'] +
        -2.0 * df['density'] +
        0.5 * df['pH'] +
        0.8 * df['sulphates'] +
        0.3 * df['alcohol'] +
        np.random.normal(0, 0.5, n_samples)  # noise
    )
    
    # Scale to 3-8 range (real wine quality scores)
    df['quality'] = np.clip(quality, 3, 8)
    
    return df

if __name__ == "__main__":
    df = generate_wine_data()
    df.to_csv('data/wine_quality.csv', index=False)
    print("✅ Data generated: data/wine_quality.csv")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Quality range: {df['quality'].min():.1f} - {df['quality'].max():.1f}")
    print(f"📈 Sample quality: {df['quality'].head(3).values}")