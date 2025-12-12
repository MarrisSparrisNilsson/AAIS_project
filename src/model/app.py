import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from unsloth import FastVisionModel
import uvicorn

# Setup MLflow
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{os.path.abspath('mlflow.db')}"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Global model variable
model = None

def get_last_run_id(experiment_name="qwen3-vl-invoice-finetune"):
    """Get the latest run ID from MLflow experiment."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise ValueError("No runs found in experiment")
    return runs[0].info.run_id

def load_model():
    """Load MLflow model once at startup."""
    global model
    print("Loading MLflow model...")
    try:
        from mlflow.tracking import MlflowClient
        import os
        import yaml
        client = MlflowClient()
        
        # FIRST: Try to find any available model in the models registry
        print("Searching for available models in registry...")
        models_dir = "mlruns/2/models"
        available_models = []
        
        if os.path.exists(models_dir):
            for model_id in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_id, "artifacts", "MLmodel")
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'r') as f:
                            mlmodel_data = yaml.safe_load(f)
                            artifact_path = os.path.join(models_dir, model_id, "artifacts")
                            created = mlmodel_data.get('utc_time_created', '')
                            available_models.append({
                                'model_id': model_id,
                                'path': artifact_path,
                                'created': created,
                                'run_id': mlmodel_data.get('run_id', '')
                            })
                    except Exception:
                        pass
        
        # If we found models, use the most recent one
        if available_models:
            available_models.sort(key=lambda x: x.get('created', ''), reverse=True)
            latest_model = available_models[0]
            print(f"Found {len(available_models)} model(s) in registry")
            print(f"   Using most recent: {latest_model['model_id']} (created: {latest_model['created']})")
            print(f"   Loading from: {latest_model['path']}")
            try:
                loaded_model = mlflow.pyfunc.load_model(latest_model['path'])
                model = loaded_model
                print(f"Model loaded successfully from registry!")
                return model
            except Exception as e:
                print(f"   Failed to load from registry: {e}")
        
        # FALLBACK: Try to get from experiment
        exp = client.get_experiment_by_name("qwen3-vl-invoice-finetune")
        if exp is None:
            raise ValueError("Experiment 'qwen3-vl-invoice-finetune' not found and no models in registry")
        
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs:
            raise ValueError("No runs found in experiment and no models in registry")
        
        run_id = runs[0].info.run_id
        experiment_id = exp.experiment_id
        print(f"Trying run ID from database: {run_id} in experiment: {experiment_id}")
        
        # Try to find model in run directory
        possible_exp_dirs = [str(experiment_id), "1", "2", "0"]
        run_dir_found = None
        
        for exp_dir in possible_exp_dirs:
            test_dir = f"mlruns/{exp_dir}/{run_id}"
            if os.path.exists(test_dir):
                run_dir_found = test_dir
                print(f"Found run directory: {run_dir_found}")
                break
        
        if run_dir_found:
            # Check for model artifacts
            artifacts_dir = os.path.join(run_dir_found, "artifacts")
            if os.path.exists(artifacts_dir):
                for item in os.listdir(artifacts_dir):
                    item_path = os.path.join(artifacts_dir, item)
                    if os.path.isdir(item_path):
                        mlmodel = os.path.join(item_path, "MLmodel")
                        if os.path.exists(mlmodel):
                            print(f"Found model at: {item}")
                            model_uri = f"runs:/{run_id}/{item}"
                            loaded_model = mlflow.pyfunc.load_model(model_uri)
                            model = loaded_model
                            print(f"Model loaded successfully!")
                            return model
        
        raise ValueError(f"Could not find model artifact in run {run_id}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_model_via_registry(model_name="qwen3vl-finetuned"):
    """Load model from MLflow Model Registry."""
    global model
    model_version = "latest"
    print(f"Loading model '{model_name}' from MLflow Model Registry...")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        print("Model loaded successfully from registry!")
        return model
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_base_model():
    """Load the base (non-finetuned) Qwen model."""
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

class BaseVisionPyFunc:
    """
    Pyfunc-like wrapper exposing predict(DataFrame) for the base Unsloth Qwen3-VL model.
    Returns a DataFrame with a 'prediction' column, matching mlflow.pyfunc models.
    """
    def __init__(self, base_model, base_tokenizer):
        import torch
        self.model = base_model.eval()
        self.tokenizer = base_tokenizer
        try:
            self.device = next(self.model.parameters()).device
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        import torch
        from PIL import Image
        results = []
        for _, row in df.iterrows():
            img = row["image"]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            instruction = row.get(
                "instruction",
                "Read the OCR in the image. Extract the invoice number, date, and total gross amount. Do not extract the currency unit for the total amount. Ensure that your output has no spaces between numbers or letters. Provide the output in the following format: JSON with keys 'invoice_nr', 'date', and 'total_amount'. All values should be represented as strings, not numbers. Provide only the requested JSON in your response."
            )
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction}]}]
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.tokenizer(img, input_text, return_tensors="pt")
            try:
                inputs = inputs.to(self.device)
            except Exception:
                pass
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    use_cache=True,
                    temperature=0.5,
                    min_p=0.1,
                )
            gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
            decoded = self.tokenizer.batch_decode(gen_trimmed, skip_special_tokens=True)
            results.append(decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded))
        return pd.DataFrame({"prediction": results})

@asynccontextmanager
async def lifespan(app):
    """Load model on startup, cleanup on shutdown."""
    global model
    print("=" * 50)
    print("Starting FastAPI server...")
    print("=" * 50)
    try:
        base_m, base_t = load_base_model()
        model = BaseVisionPyFunc(base_m, base_t)
        #load_model() # Load finetuned model
        print("=" * 50)
        print("Server ready! Model loaded successfully.")
        print("=" * 50)
    except Exception as e:
        print("=" * 50)
        print(f"CRITICAL: Failed to load model: {e}")
        print("Server will start but predictions will fail.")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        model = None
    yield
    print("Shutting down...")

app = FastAPI(title="Invoice Extraction API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Invoice Extraction API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

class PredictionResponse(BaseModel):
    prediction: str
    parsed_json: Optional[dict] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict_invoice(file: UploadFile = File(...)):
    """
    Extract invoice information from uploaded image.
    
    - **file**: Image file (PNG, JPG, etc.)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "image": temp_path,
            "instruction": "Read the OCR in the image. Extract the invoice number, date, and total gross amount. Do not extract the currency unit for the total amount. Ensure that your output has no spaces between numbers or letters. Provide the output in the following format: JSON with keys 'invoice_nr', 'date', and 'total_amount'. All values should be represented as strings, not numbers. Provide only the requested JSON in your response."
        }])
        
        # Get prediction
        preds = model.predict(input_df)
        raw_pred = str(preds.iloc[0]["prediction"])
        
        # Try to parse JSON from prediction
        parsed_json = None
        try:
            start = raw_pred.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(raw_pred)):
                    if raw_pred[i] == "{":
                        depth += 1
                    elif raw_pred[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = raw_pred[start : i + 1]
                            parsed_json = json.loads(json_str)
                            break
        except Exception:
            pass
        
        return PredictionResponse(
            prediction=raw_pred,
            parsed_json=parsed_json
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict/path")
async def predict_invoice_path(image_path: str):
    """
    Extract invoice information from image path.
    
    - **image_path**: Path to image file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "image": image_path,
            "instruction": "Read the OCR in the image. Extract the invoice number, date, and total gross amount. Do not extract the currency unit for the total amount. Ensure that your output has no spaces between numbers or letters. Provide the output in the following format: JSON with keys 'invoice_nr', 'date', and 'total_amount'. All values should be represented as strings, not numbers. Provide only the requested JSON in your response."
        }])
        
        # Get prediction
        preds = model.predict(input_df)
        raw_pred = str(preds.iloc[0]["prediction"])
        
        # Try to parse JSON from prediction
        parsed_json = None
        try:
            start = raw_pred.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(raw_pred)):
                    if raw_pred[i] == "{":
                        depth += 1
                    elif raw_pred[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = raw_pred[start : i + 1]
                            parsed_json = json.loads(json_str)
                            break
        except Exception:
            pass
        
        return PredictionResponse(
            prediction=raw_pred,
            parsed_json=parsed_json
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

