import gradio as gr
import os
import json
import requests

# FastAPI backend URL (use service name when in Docker)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

approved_invoices = []

def extract_info_via_api(image_file):
    """Extract invoice info by uploading to FastAPI backend."""
    if image_file is None:
        return None, ""
    
    try:
        # Upload file to FastAPI /predict endpoint
        with open(image_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files=files
            )
        
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
        return "Invoice approved!", approved_list
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
    gr.Markdown(f"Backend: {BACKEND_URL}")
    
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
    
    # Connect extract button to FastAPI backend
    extract_btn.click(
        fn=extract_info_via_api,
        inputs=image_input,
        outputs=[prediction_display, prediction_state]
    )
    
    # Connect approve button
    approve_btn.click(
        fn=approve_invoice,
        inputs=prediction_state,
        outputs=[status_msg, approved_list_state]
    )
    
    # Connect refresh button
    refresh_btn.click(
        fn=get_approved_list,
        outputs=approved_list_display
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)