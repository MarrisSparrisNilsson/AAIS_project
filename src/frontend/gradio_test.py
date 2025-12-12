import json
import os
from pathlib import Path

import gradio as gr
import mlflow
import pandas as pd
import requests
from invoice_inventory import display_invoice_list_layout
from mlflow.tracking import MlflowClient

# FastAPI backend URL (use service name when in Docker)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

MLFLOW_MODEL_NAME = "qwen3vl_finetuned_extraction"

approved_invoices = []


def extract_info_via_api(image_file):
    if False:
        """Extract invoice info by uploading to FastAPI backend."""
        if image_file is None:
            return None, ""

        try:
            # Upload file to FastAPI /predict endpoint
            with open(image_file, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{BACKEND_URL}/predict", files=files)

            if response.status_code != 200:
                return f"Error: {response.status_code} - {response.text}", ""

            result = response.json()
            raw_pred = result.get("prediction", "")
            parsed_json = result.get("parsed_json", {})

            if parsed_json:
                display_text = json.dumps(parsed_json, indent=2)
                state_json = json.dumps(parsed_json)
            else:
                display_text = raw_pred
                state_json = raw_pred

            return display_text, state_json

        except Exception as e:
            return f"Error connecting to backend: {str(e)}", ""
    else:
        json_example = {
            "invoice": "INV-1001",
            "date": "2023-10-01",
            "client": "ABC Supplies",
            "seller": "Hopkins",
            "total": "$1,500.00",
            "line_items": [
                {"description": "Item A", "quantity": 2, "unit_price": "$500.00", "total": "$1,000.00"},
                {"description": "Item B", "quantity": 1, "unit_price": "$500.00", "total": "$500.00"},
            ],
        }

        return json.dumps(json_example, indent=2), json.dumps(json_example)


def approve_invoice(prediction_state):
    """Approve and store the invoice prediction."""
    if not prediction_state or prediction_state.strip() == "":
        return "Please extract invoice info first.", ""

    try:
        invoice_data = json.loads(prediction_state)
        approved_invoices.append(invoice_data)

        approved_list = "\n".join([f"Invoice {i+1}: {json.dumps(inv)}" for i, inv in enumerate(approved_invoices)])
        append_to_json_array(invoice_data, filename="approved_invoices_v1.json")
        append_to_json_file(invoice_data, filename="approved_invoices_v2.json")

        return "Invoice approved!", approved_list
    except Exception as e:
        return f"Error approving invoice: {str(e)}", ""


def append_to_json_file(data, filename="approved_invoices.json"):
    """
    Appends a JSON object to filename (JSONL format)
    in the same directory as this script.
    """
    # Directory where the current script is located
    CURRENT_DIR = Path(__file__).parent.resolve()
    path = CURRENT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f)
        f.write("\n")


def append_to_json_array(data, filename="approved_invoices.json"):
    CURRENT_DIR = Path(__file__).parent.resolve()
    path = CURRENT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            json.dump([data], f, indent=2)
        return

    with path.open("r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(data)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()


def get_approved_list():
    """Fetch current approved invoices list."""

    if not approved_invoices:
        return "No approved invoices yet."
    return "\n".join([f"Invoice {i+1}: {json.dumps(inv)}" for i, inv in enumerate(approved_invoices)])


with gr.Blocks(title="Invoice Extractor") as demo:
    gr.Markdown("# Invoice Information Extractor")
    # gr.Markdown(f"Backend: {BACKEND_URL}")

    prediction_state = gr.State(value="")
    approved_list_state = gr.State(value="")

    with gr.Tabs():
        # Tab 1: Extraction
        with gr.Tab("Extract Invoice"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Files(file_types=["image"], label="Upload Invoice Image")
                    # image_input.upload(fn=None, inputs=[], outputs=[], trigger_mode="multiple")
                    extract_btn = gr.Button("Extract Information", variant="primary")

                with gr.Column():
                    prediction_display = gr.Textbox(label="Extracted Information", lines=6, interactive=False)

            with gr.Row():
                approve_btn = gr.Button("Approve", variant="success")
                deny_btn = gr.Button("Deny", variant="danger")
                status_msg = gr.Textbox(label="Status", interactive=False)

        # Tab 2: Approved List
        with gr.Tab("Approved Invoices"):
            approved_list_display = gr.Textbox(label="List of Approved Invoices", lines=15, interactive=False)
            refresh_btn = gr.Button("Refresh", variant="secondary")

        # Tab 3: Invoice Inventory Browser
        with gr.Tab("Invoice Inventory Browser"):
            display_invoice_list_layout()

    # Connect extract button to FastAPI backend
    extract_btn.click(fn=extract_info_via_api, inputs=image_input, outputs=[prediction_display, prediction_state])

    # Connect approve button
    approve_btn.click(fn=approve_invoice, inputs=prediction_state, outputs=[status_msg, approved_list_state])

    # Connect refresh button
    refresh_btn.click(fn=get_approved_list, outputs=approved_list_display)

if __name__ == "__main__":
    demo.launch()
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
