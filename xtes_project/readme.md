# Make sure you're in the conda environment
conda activate donat

# Terminal 1: Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: run the code and 
python src/train_model.py



















# used BentoML to containerize the model
- Serve models to other applications
- Handle high traffic
- Scale automatically
- Deploy to production environments