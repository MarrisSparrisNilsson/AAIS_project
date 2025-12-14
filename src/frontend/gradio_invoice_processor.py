import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import gradio as gr
from fastapi import requests

# Path to store approved invoices
APPROVED_INVOICES_FILE = "approved_invoices.json"

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

MLFLOW_MODEL_NAME = "qwen3vl_finetuned_extraction"


class InvoiceProcessor:
    def __init__(self):
        self.processed_invoices = []
        self.approved_invoices = self.load_approved_invoices()

    def load_approved_invoices(self) -> List[Dict]:
        """Load approved invoices from JSON file"""
        file_path = Path(__file__).parent / APPROVED_INVOICES_FILE
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []

    def save_approved_invoices(self):
        """Save approved invoices to JSON file"""
        with open(Path(__file__).parent / APPROVED_INVOICES_FILE, "w") as f:
            json.dump(self.approved_invoices, f, indent=2)

    async def extract_information_from_image(self, image_path: str, index: int) -> Dict:
        if True:
            """Extract invoice info by uploading to FastAPI backend."""
            if image_path is None:
                return None, ""

            try:
                # Upload file to FastAPI /predict endpoint
                with open(image_path, "rb") as f:
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

                # Output should have invoice, date, client, seller, total
                return json.dumps(display_text, indent=2), json.dumps(state_json)

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

        # """
        # Simulate VLM model extraction - Replace this with your actual VLM model
        # """
        # # Simulate processing time
        # await asyncio.sleep(1.5 + index * 0.3)

        # # Mock extracted data - replace with actual VLM inference
        # extracted_data = {
        #     "invoice_nr": f"0909043{index}",
        #     "date": "01/23/2021",
        #     "seller": "Hopkins",
        #     "client": "Sims PLC",
        #     "total": f"${640.12 - index * 100:.2f}",
        #     "items": [
        #         {"description": "Professional Services", "quantity": 1, "price": 350.00},
        #         {"description": "Consultation Fee", "quantity": 2, "price": 95.06},
        #     ],
        #     "address": "123 Main St, City, State",
        #     "tax": "10%",
        # }

        # return {
        #     "id": f"invoice_{int(time.time())}_{index}",
        #     "file_path": image_path,
        #     "file_name": Path(image_path).name,
        #     "extracted_data": extracted_data,
        #     "checked": False,
        # }


# Initialize processor
processor = InvoiceProcessor()


def create_invoice_list_html():
    """Create HTML list of invoices with checkboxes"""
    if not processor.processed_invoices:
        return "<div style='text-align: center; padding: 40px; color: #999;'>No invoices processed yet</div>"

    html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"

    for idx, invoice in enumerate(processor.processed_invoices):
        checked = "checked" if invoice.get("checked", False) else ""

        html += f"""
        <div class='invoice-card' style='border: 1px solid #ddd; border-radius: 6px; padding: 16px; background: white;
                    display: flex; align-items: center; gap: 16px;'>
            <div style='flex: 1; display: flex; align-items: center; gap: 12px; cursor: pointer;'
                onclick='document.getElementById("invoice_selector").value = "{idx}";
                        document.getElementById("invoice_selector").dispatchEvent(new Event("change"));'>
                {invoice['file_name']}
            </div>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 4px;'>
                <input type='checkbox' {checked} class='invoice-checkbox'
                    onchange='
                        fetch("/gradio_api/call/toggle_checkbox", {{
                            method: "POST",
                            headers: {{"Content-Type": "application/json"}},
                            body: JSON.stringify({{data: [{idx}]}})
                        }}).then(() => {{
                            fetch("/gradio_api/call/refresh_list", {{
                                method: "POST",
                                headers: {{"Content-Type": "application/json"}},
                                body: JSON.stringify({{data: []}})
                            }}).then(r => r.json()).then(result => {{
                                    document.querySelector("#invoice_list iframe").contentDocument.body.innerHTML = result.data[0];
                            }});
                        }});
                        
                        // Check if any checkbox is checked and enable/disable buttons
                        const anyChecked = Array.from(document.querySelectorAll(".invoice-checkbox")).some(cb => cb.checked);
                        document.querySelectorAll(".approve-btn").forEach(btn => {{
                            btn.disabled = !anyChecked;
                            btn.style.opacity = anyChecked ? "1" : "0.5";
                            btn.style.cursor = anyChecked ? "pointer" : "not-allowed";
                        }});
                    '
                    style='width: 18px; height: 18px; cursor: pointer;'>
                    <span style='font-size: 11px; color: #666;'>Select</span>
            </div>
        </div>
        """

    html += "</div>"
    return html


def create_approved_list_html(search_query: str = ""):
    """Create HTML list of approved invoices"""
    invoices = processor.approved_invoices

    if search_query:
        invoices = [inv for inv in invoices if search_query.lower() in json.dumps(inv["extracted_data"]).lower()]

    if not invoices:
        return "<div style='text-align: center; padding: 40px; color: #999;'>No approved invoices found</div>"

    html = f"""
    <div style='text-align: center; font-weight: 600; margin-bottom: 16px; font-size: 16px;'>
        Listed Invoices ({len(invoices)})
    </div>
    <div style='display: flex; flex-direction: column; gap: 12px;'>
    """

    for idx, invoice in enumerate(invoices):
        data = invoice["extracted_data"]

        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 6px; padding: 20px; background: white;
                    cursor: pointer; transition: background 0.2s;'
                onmouseover='this.style.background="#f9fafb"'
                onmouseout='this.style.background="white"'
                onclick='document.getElementById("approved_selector").value = "{idx}";
                        document.getElementById("approved_selector").dispatchEvent(new Event("change"));'>
            <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px;'>
                <div>
                    <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Invoice nr:</div>
                    <div style='font-weight: 600;'>{data['invoice_nr']}</div>
                </div>
                <div>
                    <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Date:</div>
                    <div style='font-weight: 600;'>{data['date']}</div>
                </div>
                <div>
                    <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Seller:</div>
                    <div style='font-weight: 600;'>{data['seller']}</div>
                </div>
                <div>
                    <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Client:</div>
                    <div style='font-weight: 600;'>{data['client']}</div>
                </div>
                <div>
                    <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Total:</div>
                    <div style='font-weight: 600;'>{data['total']}</div>
                </div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def extract_information(files):
    """Extract information from uploaded files"""

    if not files:
        return create_invoice_list_html()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    processor.processed_invoices = []

    if files is not list:
        idx = 0
        # TODO: Replace with actual VLM model inference that extracts invoice data
        result = loop.run_until_complete(processor.extract_information_from_image(files, idx))
        processor.processed_invoices.append(result)

        view_extraction_invoice(idx)
    else:
        for idx, file in enumerate(files):
            # TODO: Replace with actual VLM model inference that extracts invoice data
            result = loop.run_until_complete(processor.extract_information_from_image(file, idx))
            processor.processed_invoices.append(result)

    loop.close()
    return create_invoice_list_html()


def mark_all_checked():
    """Mark all invoices as checked"""
    for inv in processor.processed_invoices:
        inv["checked"] = True
    return "Unmark all"  # Return count, enable buttons, and new label


def unmark_all_checked():
    """Unmark all invoices"""
    for inv in processor.processed_invoices:
        inv["checked"] = False
    return "Mark all"  # Return count, disable buttons, and new label


def toggle_mark_all(current_label):
    """Toggle between mark all and unmark all"""
    if current_label == "Mark all":
        return mark_all_checked()
    else:
        return unmark_all_checked()


def toggle_checkbox(idx: int):
    """Toggle checkbox state"""
    if 0 <= idx < len(processor.processed_invoices):
        processor.processed_invoices[idx]["checked"] = not processor.processed_invoices[idx].get("checked", False)
    return create_invoice_list_html()


def refresh_list():
    """Refresh the invoice list"""
    return create_invoice_list_html()


def approve_invoices():
    """Approve checked invoices and save to JSON file"""
    checked_invoices = [inv for inv in processor.processed_invoices if inv.get("checked", False)]

    if checked_invoices:
        # Append to approved list
        processor.approved_invoices.extend(checked_invoices)
        processor.save_approved_invoices()

        # Remove approved invoices from processed list
        processor.processed_invoices = [inv for inv in processor.processed_invoices if not inv.get("checked", False)]

        print(f"✓ Approved {len(checked_invoices)} invoice(s) and saved to {APPROVED_INVOICES_FILE}")

    return (
        create_invoice_list_html(),
        create_approved_list_html(),
        gr.Tabs(selected=1),  # Switch to approved tab
        None,
        None,
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def deny_invoices():
    """Clear all processed invoices"""
    processor.processed_invoices = []
    return (create_invoice_list_html(), None, None, None)


def view_extraction_invoice(idx: str):
    """View details for extraction invoice"""
    try:
        idx = int(idx)
        if 0 <= idx < len(processor.processed_invoices):
            invoice = processor.processed_invoices[idx]
            return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
    except:
        pass
    return None, "", gr.update(visible=False)


def view_approved_invoice(idx: str):
    """View details for approved invoice"""
    try:
        idx = int(idx)
        if 0 <= idx < len(processor.approved_invoices):
            invoice = processor.approved_invoices[idx]
            return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
    except:
        pass
    return None, "", gr.update(visible=False)


def search_approved(query: str):
    """Search approved invoices"""
    return create_approved_list_html(query)


# Create Gradio Interface
with gr.Blocks() as demo:
    mark_all_label = gr.State("Mark all")

    gr.Markdown("# Invoice Processing System")

    with gr.Tabs() as tabs:
        # Extraction Tab
        with gr.Tab("Information Extraction", id=0):
            gr.Markdown("## Upload and Extract Invoice Information")

            file_upload = gr.File(
                label="Upload image files (Component)", file_count="single", type="filepath", file_types=["image"]
            )

            with gr.Row():
                extract_btn = gr.Button("Extract information", variant="secondary", scale=4)
                mark_all_btn = gr.Button("Mark all", scale=1)

            gr.Markdown("### Processed Invoices")

            invoice_list = gr.HTML(value=create_invoice_list_html(), elem_id="invoice_list")

            # Hidden selector for clicking invoices
            invoice_selector = gr.Textbox(visible=False, elem_id="invoice_selector")

            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary", interactive=False, elem_classes="approve-btn")
                deny_btn = gr.Button("Deny", variant="secondary", interactive=False, elem_classes="approve-btn")

            # Modal for extraction view
            with gr.Column(visible=True) as extraction_modal:
                gr.Markdown("## Invoice Preview")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Image")
                        extraction_modal_image = gr.Image(label="", type="filepath")

                    with gr.Column():
                        gr.Markdown("### JSON Preview")
                        extraction_modal_json = gr.Code(label="", language="json", lines=25)

        # Approved Tab
        with gr.Tab("Approved Invoices", id=1):
            gr.Markdown("## Approved Invoices")

            with gr.Column():
                gr.HTML("<h3>Search and browse invoices.</h3>")
                search_box = gr.Textbox(placeholder="Hopkins", scale=2, label="Search Invoices", show_label=False)

            approved_list = gr.HTML(value=create_approved_list_html(), elem_id="approved_list")

            # Hidden selector for clicking approved invoices
            approved_selector = gr.Textbox(visible=False, elem_id="approved_selector")

            # Modal for approved view
            with gr.Column(visible=False) as approved_modal:
                gr.Markdown("## Invoice Preview")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Image")
                        approved_modal_image = gr.Image(label="", type="filepath")

                    with gr.Column():
                        gr.Markdown("### JSON Preview")
                        approved_modal_json = gr.Code(label="", language="json", lines=20)

    # Event handlers for Extraction Tab
    extract_btn.click(fn=extract_information, inputs=[file_upload], outputs=[invoice_list]).then(
        fn=lambda: (
            view_extraction_invoice("0")
            if len(processor.processed_invoices) > 0
            else (None, "", gr.update(visible=False))
        ),
        inputs=[],
        outputs=[extraction_modal_image, extraction_modal_json, extraction_modal],
    )

    mark_all_btn.click(
        fn=toggle_mark_all,
        inputs=[mark_all_label],
        outputs=[mark_all_label],
        js="""
        () => {
            document.querySelectorAll('.invoice-checkbox').forEach(cb => cb.checked = !cb.checked);
            document.querySelectorAll('.approve-btn').forEach(btn => btn.disabled = !btn.disabled);
        }""",
    ).then(fn=lambda label: gr.update(value=label), inputs=[mark_all_label], outputs=[mark_all_btn])

    invoice_selector.change(
        fn=view_extraction_invoice,
        inputs=[invoice_selector],
        outputs=[extraction_modal_image, extraction_modal_json, extraction_modal],
    )

    approve_btn.click(
        fn=approve_invoices,
        inputs=[],
        outputs=[
            invoice_list,
            approved_list,
            tabs,
            file_upload,
            extraction_modal_image,
            extraction_modal_json,
            approve_btn,
            deny_btn,
        ],
    )

    deny_btn.click(
        fn=deny_invoices, inputs=[], outputs=[invoice_list, file_upload, extraction_modal_image, extraction_modal_json]
    )

    # Event handlers for Approved Tab
    search_box.change(fn=search_approved, inputs=[search_box], outputs=[approved_list])

    approved_selector.change(
        fn=view_approved_invoice,
        inputs=[approved_selector],
        outputs=[approved_modal_image, approved_modal_json, approved_modal],
    )

    # API endpoints for checkbox toggling
    toggle_checkbox_api = gr.Number(visible=False)
    toggle_checkbox_api.change(
        fn=toggle_checkbox,
        inputs=[toggle_checkbox_api],
        outputs=[invoice_list],
        api_name="toggle_checkbox",
        js="""
        () => {
           document.querySelectorAll('.approve-btn').forEach(btn => btn.disabled = btn.disabled ? True : false);
        }""",
    )

    # API endpoint for refreshing invoice list
    refresh_list_api = gr.Button(visible=False)
    refresh_list_api.click(fn=refresh_list, inputs=[], outputs=[invoice_list], api_name="refresh_list")


if __name__ == "__main__":
    demo.launch(
        share=False,
        css="""
        .container { max-width: 1200px; margin: auto; }
        """,
    )
