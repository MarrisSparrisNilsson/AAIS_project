import gradio as gr
import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{os.path.abspath('mlflow.db')}"
mlflow.set_tracking_uri(mlflow_tracking_uri)

approved_invoices = []

def get_last_run_id(experiment_name="qwen3-vl-invoice-finetune"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        return None
    return runs[0].info.run_id

run_id = get_last_run_id()
model_uri = f"runs:/{run_id}/qwen3vl_finetuned_extraction"
model = mlflow.pyfunc.load_model(model_uri)

def extract_info(image_path):
    """Extract invoice info from image."""
    if image_path is None:
        return None, ""
    
    input_df = pd.DataFrame([{
        "image": image_path,
        "instruction": "Read the OCR in the image. Extract the invoice number, date, and total amount. Provide the output in the following format: JSON with keys 'invoice_nr', 'date', and 'total_amount'."
    }])
    
    preds = model.predict(input_df)
    raw_pred = str(preds.iloc[0]["prediction"])
    
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
                        parsed = json.loads(json_str)
                        return json.dumps(parsed, indent=2), json.dumps(parsed)
    except Exception as e:
        pass
    
    return raw_pred, raw_pred

def approve_invoice(prediction_state):
    """Approve and store the invoice prediction."""
    if not prediction_state or prediction_state.strip() == "":
        return "Please extract invoice info first.", ""
    
    try:
        invoice_data = json.loads(prediction_state)
        approved_invoices.append(invoice_data)
        
        approved_list = "\n".join([
            f"Invoice {i+1}: {json.dumps(inv)}"
            for i, inv in enumerate(approved_invoices)
        ])
        return "✓ Invoice approved!", approved_list
    except Exception as e:
        return f"Error approving invoice: {str(e)}", ""

def get_approved_list():
    """Fetch current approved invoices list."""
    if not approved_invoices:
        return "No approved invoices yet."
    return "\n".join([
        f"Invoice {i+1}: {json.dumps(inv)}"
        for i, inv in enumerate(approved_invoices)
    ])

with gr.Blocks(title="Invoice Extractor") as demo:
    gr.Markdown("# Invoice Information Extractor")
    
    prediction_state = gr.State(value="")
    approved_list_state = gr.State(value="")
    
    with gr.Tabs():
        # Tab 1: Extraction
        with gr.Tab("Extract Invoice"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="filepath", label="Upload Invoice Image")
                    extract_btn = gr.Button("Extract Information", variant="primary")
                
                with gr.Column():
                    prediction_display = gr.Textbox(label="Extracted Information", lines=6, interactive=False)
            
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="success")
                status_msg = gr.Textbox(label="Status", interactive=False)
        
        # Tab 2: Approved List
        with gr.Tab("Approved Invoices"):
            approved_list_display = gr.Textbox(label="List of Approved Invoices", lines=15, interactive=False)
            refresh_btn = gr.Button("Refresh", variant="secondary")
    
    # Connect extract button
    extract_btn.click(
        fn=extract_info,
        inputs=image_input,
        outputs=[prediction_display, prediction_state]
    )
    
    # Connect approve button
    approve_btn.click(
        fn=approve_invoice,
        inputs=prediction_state,
        outputs=[status_msg, approved_list_state]
    )
    
    # Connect refresh button on approved list tab
    refresh_btn.click(
        fn=get_approved_list,
        outputs=approved_list_display
    )

if __name__ == "__main__":
    demo.launch()