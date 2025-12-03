import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{os.path.abspath('mlflow.db')}"
mlflow.set_tracking_uri(mlflow_tracking_uri)

def get_last_run_id(experiment_name="qwen3-vl-invoice-finetune"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        return None
    return runs[0].info.run_id

def extract_info(image_path):
    # Load the MLflow model
    run_id = get_last_run_id()
    model_uri = f"runs:/{run_id}/qwen3vl_finetuned_extraction"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Prepare input DataFrame
    input_df = pd.DataFrame([{
        "image": image_path,
        "instruction": "Read the OCR in the image. Extract the invoice number, date, and total amount. Provide the output in the following format: JSON with keys 'invoice_nr', 'date', and 'total_amount'."
    }])
    
    # Get predictions
    preds = model.predict(input_df)
    
    return preds.iloc[0]

if __name__ == "__main__":
    # Example usage
    test_image_path = "invoices_dataset/mini_dataset/images/test/dataset1_katanaml_test_katanaml_0004.png"
    extracted_info = extract_info(test_image_path)
    print(extracted_info["prediction"])