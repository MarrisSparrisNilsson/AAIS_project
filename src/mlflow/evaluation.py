import json
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from PIL import Image
import torch
from unsloth import FastVisionModel
import yaml

def load_eval_data(data_dir: str, limit: int = None) -> pd.DataFrame:
    """Load evaluation data from JSON files and images."""
    eval_data = []
    data_path = Path(data_dir)
    
    json_files = sorted(data_path.glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    
    for json_file in json_files:
        with open(json_file, "r") as f:
            ground_truth = json.load(f)
        
        # Assume image has same name as JSON
        image_path = json_file.with_suffix(".png")
        if not image_path.exists():
            image_path = json_file.with_suffix(".jpg")
        
        if image_path.exists():
            eval_data.append({
                "image_path": str(image_path),
                "invoice_nr_gt": ground_truth.get("invoice_nr", ""),
                "date_gt": ground_truth.get("date", ""),
                "total_amount_gt": ground_truth.get("total_amount", "")
            })
    
    return pd.DataFrame(eval_data)


def load_base_model():
    """Load the base (non-finetuned) Qwen model."""
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model(verbose: bool = False):
    """Load MLflow finetuned model (same logic as app.py)."""
    if verbose:
        print("Loading MLflow finetuned model...")
    try:
        client = MlflowClient()
        
        # FIRST: Try to find any available model in the models registry
        if verbose:
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
            if verbose:
                print(f"Found {len(available_models)} model(s) in registry")
                print(f"   Using most recent: {latest_model['model_id']} (created: {latest_model['created']})")
                print(f"   Loading from: {latest_model['path']}")
            try:
                loaded_model = mlflow.pyfunc.load_model(latest_model['path'])
                if verbose:
                    print(f"Model loaded successfully from registry!")
                return loaded_model
            except Exception as e:
                if verbose:
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
        if verbose:
            print(f"Trying run ID from database: {run_id} in experiment: {experiment_id}")
        
        # Try to find model in run directory
        possible_exp_dirs = [str(experiment_id), "1", "2", "0"]
        run_dir_found = None
        
        for exp_dir in possible_exp_dirs:
            test_dir = f"mlruns/{exp_dir}/{run_id}"
            if os.path.exists(test_dir):
                run_dir_found = test_dir
                if verbose:
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
                            if verbose:
                                print(f"Found model at: {item}")
                            model_uri = f"runs:/{run_id}/{item}"
                            loaded_model = mlflow.pyfunc.load_model(model_uri)
                            if verbose:
                                print(f"Model loaded successfully!")
                            return loaded_model
        
        raise ValueError(f"Could not find model artifact in run {run_id}")
        
    except Exception as e:
        print(f"Error loading finetuned model: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_prediction(model, tokenizer, image_path: str, instruction: str) -> str:
    """Generate prediction for a single image."""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=400,
            use_cache=True,
            temperature=0.5,
            min_p=0.1,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)
    ]
    
    decoded = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return decoded[0] if isinstance(decoded, (list, tuple)) else decoded


def generate_prediction_pyfunc(model, image_path: str, instruction: str) -> str:
    """Generate prediction using MLflow pyfunc model."""
    input_df = pd.DataFrame([{
        "image": image_path,
        "instruction": instruction
    }])
    pred = model.predict(input_df)
    return pred["prediction"].iloc[0]


def parse_prediction(pred_text: str, verbose: bool = False) -> dict:
    """Extract JSON from prediction text."""
    try:
        # Try to find JSON in the response
        start = pred_text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(pred_text)):
                if pred_text[i] == "{":
                    depth += 1
                elif pred_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = pred_text[start : i + 1]
                        return json.loads(json_str)
    except Exception as e:
        if verbose:
            print(f"Error parsing JSON: {e}")
            print(f"Raw text: {pred_text}")
    
    return {"invoice_nr": "", "date": "", "total_amount": ""}


def calculate_metrics(eval_df: pd.DataFrame) -> dict:
    """Calculate accuracy metrics for each field."""
    metrics = {}
    
    # Invoice number exact match
    invoice_match = (eval_df["invoice_nr_pred"] == eval_df["invoice_nr_gt"]).sum()
    metrics["invoice_nr_accuracy"] = invoice_match / len(eval_df)
    
    # Date exact match
    date_match = (eval_df["date_pred"] == eval_df["date_gt"]).sum()
    metrics["date_accuracy"] = date_match / len(eval_df)
    
    # Total exact match
    total_match = (eval_df["total_amount_pred"] == eval_df["total_amount_gt"]).sum()
    metrics["total_amount_accuracy"] = total_match / len(eval_df)
    
    # Overall accuracy (all fields correct)
    all_correct = (
        (eval_df["invoice_nr_pred"] == eval_df["invoice_nr_gt"]) &
        (eval_df["date_pred"] == eval_df["date_gt"]) &
        (eval_df["total_amount_pred"] == eval_df["total_amount_gt"])
    ).sum()
    metrics["overall_accuracy"] = all_correct / len(eval_df)
    
    return metrics


def evaluate_models(finetuned_model=None, limit: int = 5, verbose: bool = False):
    """Compare finetuned vs non-finetuned Qwen models."""
    
    # Set MLflow tracking
    mlflow_dir = Path("./").resolve()
    db_path = mlflow_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("qwen3-vl-invoice-evaluation")
    
    data_dir = "../../invoices_dataset/eval_data"
    eval_data = load_eval_data(data_dir, limit=limit)
    
    print(f"\n{'='*80}")
    print(f"Evaluating on {len(eval_data)} samples")
    print(f"{'='*80}\n")
    
    instruction = "Read the OCR in the image. Extract the invoice number, date, and total gross amount. Do not extract the currency unit for the total amount. Ensure that your output has no spaces between numbers or letters. Provide the output in the following format: JSON with keys 'invoice_nr', 'date', and 'total_amount'. All values should be represented as strings, not numbers. Provide only the requested JSON in your response."
    
    with mlflow.start_run(run_name="qwen_model_comparison"):
        # Evaluate base model
        if verbose:
            print("Loading base model...")
        base_model, base_tokenizer = load_base_model()
        
        if verbose:
            print("Generating base model predictions...\n")
        base_predictions = []
        for idx, row in eval_data.iterrows():
            if verbose:
                print(f"\n{'─'*80}")
                print(f"Sample {idx + 1}/{len(eval_data)}")
                print(f"{'─'*80}")
                print(f"Image: {row['image_path']}")
            
            pred_text = generate_prediction(base_model, base_tokenizer, row["image_path"], instruction)
            parsed = parse_prediction(pred_text, verbose=verbose)
            
            if verbose:
                print(f"\nRaw prediction:\n{pred_text}")
                print(f"\nParsed prediction:")
                print(f"  invoice_nr: '{parsed.get('invoice_nr', '')}'")
                print(f"  date: '{parsed.get('date', '')}'")
                print(f"  total_amount: '{parsed.get('total_amount', '')}'")
                
                print(f"\nGround truth:")
                print(f"  invoice_nr: '{row['invoice_nr_gt']}'")
                print(f"  date: '{row['date_gt']}'")
                print(f"  total_amount: '{row['total_amount_gt']}'")
            
            base_predictions.append({
                "invoice_nr_pred": parsed.get("invoice_nr", ""),
                "date_pred": parsed.get("date", ""),
                "total_amount_pred": parsed.get("total_amount", ""),
                "raw_prediction": pred_text
            })
        
        base_pred_df = pd.DataFrame(base_predictions)
        eval_data_base = pd.concat([eval_data, base_pred_df], axis=1)
        
        base_metrics = calculate_metrics(eval_data_base)
        print(f"\n{'='*80}")
        print(f"Base Model Metrics: {base_metrics}")
        print(f"{'='*80}\n")
        
        for metric_name, value in base_metrics.items():
            mlflow.log_metric(f"base_{metric_name}", value)
        
        # Evaluate finetuned model if provided
        if finetuned_model is not None:
            if verbose:
                print("\nGenerating finetuned model predictions...\n")
            finetuned_predictions = []
            for idx, row in eval_data.iterrows():
                if verbose:
                    print(f"\n{'─'*80}")
                    print(f"Sample {idx + 1}/{len(eval_data)} (Finetuned)")
                    print(f"{'─'*80}")
                
                pred_text = generate_prediction_pyfunc(finetuned_model, row["image_path"], instruction)
                parsed = parse_prediction(pred_text, verbose=verbose)
                
                if verbose:
                    print(f"\nRaw prediction:\n{pred_text}")
                    print(f"\nParsed prediction:")
                    print(f"  invoice_nr: '{parsed.get('invoice_nr', '')}'")
                    print(f"  date: '{parsed.get('date', '')}'")
                    print(f"  total_amount: '{parsed.get('total_amount', '')}'")

                    print(f"\nGround truth:")
                    print(f"  invoice_nr: '{row['invoice_nr_gt']}'")
                    print(f"  date: '{row['date_gt']}'")
                    print(f"  total_amount: '{row['total_amount_gt']}'")
                
                finetuned_predictions.append({
                    "invoice_nr_pred": parsed.get("invoice_nr", ""),
                    "date_pred": parsed.get("date", ""),
                    "total_amount_pred": parsed.get("total_amount", ""),
                    "raw_prediction": pred_text
                })
            
            finetuned_pred_df = pd.DataFrame(finetuned_predictions)
            eval_data_finetuned = pd.concat([eval_data, finetuned_pred_df], axis=1)
            
            finetuned_metrics = calculate_metrics(eval_data_finetuned)
            print(f"\n{'='*80}")
            print(f"Finetuned Model Metrics: {finetuned_metrics}")
            print(f"{'='*80}\n")
            
            for metric_name, value in finetuned_metrics.items():
                mlflow.log_metric(f"finetuned_{metric_name}", value)
            
            # Save comparison results
            eval_data_finetuned.to_csv("finetuned_results.csv", index=False)
            mlflow.log_artifact("finetuned_results.csv")
        
        # Save base results
        eval_data_base.to_csv("base_results.csv", index=False)
        mlflow.log_artifact("base_results.csv")


if __name__ == "__main__":
    # Set MLflow tracking first
    mlflow_dir = Path("./").resolve()
    db_path = mlflow_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    # Try to load finetuned model
    finetuned_model = None
    try:
        finetuned_model = load_finetuned_model(verbose=True)
        print("Finetuned model loaded successfully\n")
    except Exception as e:
        print(f"Could not load finetuned model: {e}")
        print("Proceeding with base model evaluation only\n")
    
    # Run evaluation with verbose=True for debug output
    evaluate_models(finetuned_model=finetuned_model, limit=100, verbose=False)