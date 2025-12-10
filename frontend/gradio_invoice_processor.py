#  =========================================

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import gradio as gr

# Path to store approved invoices
APPROVED_INVOICES_FILE = "approved_invoices.json"


class InvoiceProcessor:
    def __init__(self):
        self.processed_invoices = []
        self.approved_invoices = self.load_approved_invoices()

    def load_approved_invoices(self) -> List[Dict]:
        """Load approved invoices from JSON file"""
        if os.path.exists(APPROVED_INVOICES_FILE):
            with open(APPROVED_INVOICES_FILE, "r") as f:
                return json.load(f)
        return []

    def save_approved_invoices(self):
        """Save approved invoices to JSON file"""
        with open(Path(__file__).parent / APPROVED_INVOICES_FILE, "w") as f:
            json.dump(self.approved_invoices, f, indent=2)

    async def extract_information_from_image(self, image_path: str, index: int) -> Dict:
        """
        Simulate VLM model extraction - Replace this with your actual VLM model
        """
        # Simulate processing time
        await asyncio.sleep(1.5 + index * 0.3)

        # Mock extracted data - replace with actual VLM inference
        extracted_data = {
            "invoice_nr": f"0909043{index}",
            "date": "01/23/2021",
            "seller": "Hopkins",
            "client": "Sims PLC",
            "total": f"${640.12 - index * 100:.2f}",
            "items": [
                {"description": "Professional Services", "quantity": 1, "price": 350.00},
                {"description": "Consultation Fee", "quantity": 2, "price": 95.06},
            ],
            "address": "123 Main St, City, State",
            "tax": "10%",
        }

        return {
            "id": f"invoice_{int(time.time())}_{index}",
            "file_path": image_path,
            "file_name": Path(image_path).name,
            "extracted_data": extracted_data,
            "checked": False,
        }


# Initialize processor
processor = InvoiceProcessor()


def extract_information(files):
    """Wrapper for async processing"""
    if not files:
        return gr.update(choices=[])

    # Run async processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    processor.processed_invoices = []
    for idx, file in enumerate(files):
        result = loop.run_until_complete(processor.extract_information_from_image(file, idx))
        processor.processed_invoices.append(result)

    loop.close()

    # Return dropdown choices
    choices = [
        f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
        for i, inv in enumerate(processor.processed_invoices)
    ]
    return gr.update(choices=choices, value=None)


def mark_all():
    """Mark all invoices as checked"""
    for invoice in processor.processed_invoices:
        invoice["checked"] = True
    return get_checkboxes_state()


def get_checkboxes_state():
    """Get current checkbox states"""
    return [gr.update(value=inv.get("checked", False)) for inv in processor.processed_invoices] + [
        gr.update(visible=False)
    ] * (10 - len(processor.processed_invoices))


def toggle_checkbox_fn(idx, *checkbox_values):
    """Toggle checkbox at index"""
    if 0 <= idx < len(processor.processed_invoices):
        processor.processed_invoices[idx]["checked"] = checkbox_values[idx]
    return None


def approve_invoices(*checkbox_values):
    print(checkbox_values)
    """Approve checked invoices and save to file"""
    # Update checkbox states
    for i, checked in enumerate(checkbox_values[: len(processor.processed_invoices)]):

        if i < len(processor.processed_invoices):
            processor.processed_invoices[i]["checked"] = checked

    checked_invoices = [inv for inv in processor.processed_invoices if inv.get("checked", False)]
    print("Checked invoices:", checked_invoices)

    if checked_invoices:
        processor.approved_invoices.extend(checked_invoices)
        processor.save_approved_invoices()
        processor.processed_invoices = [inv for inv in processor.processed_invoices if not inv.get("checked", False)]

    # Update extraction view
    choices = [
        f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
        for i, inv in enumerate(processor.processed_invoices)
    ]
    print("Updated extraction choices:", choices)

    # Update approved view
    approved_choices = [
        f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
        for i, inv in enumerate(processor.approved_invoices)
    ]
    print("Updated approved choices:", approved_choices)

    # Return: extraction_dropdown + 10 checkboxes + approved_dropdown + tabs
    checkbox_updates = get_checkboxes_state()
    print("Checkbox updates:", checkbox_updates)
    return (
        [gr.update(choices=choices, value=None)]
        + checkbox_updates
        + [gr.update(choices=approved_choices, value=None), gr.Tabs(selected=1)]
    )


def deny_invoices():
    """Clear all invoices"""
    processor.processed_invoices = []
    checkbox_updates = get_checkboxes_state()
    return [gr.update(choices=[], value=None)] + checkbox_updates


def view_extraction_invoice(selection):
    """View details for extraction invoice"""
    if selection is None:
        return None, "", gr.update(visible=False)

    try:
        idx = int(selection.split(":")[0])
        if 0 <= idx < len(processor.processed_invoices):
            invoice = processor.processed_invoices[idx]
            return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
    except:
        pass

    return None, "", gr.update(visible=False)


def view_approved_invoice(selection):
    """View details for approved invoice"""
    if selection is None:
        return None, "", gr.update(visible=False)

    try:
        idx = int(selection.split(":")[0])
        if 0 <= idx < len(processor.approved_invoices):
            invoice = processor.approved_invoices[idx]
            return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
    except:
        pass

    return None, "", gr.update(visible=False)


def search_approved(query: str):
    """Search approved invoices"""
    if not query:
        choices = [
            f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
            for i, inv in enumerate(processor.approved_invoices)
        ]
    else:
        filtered = [
            (i, inv)
            for i, inv in enumerate(processor.approved_invoices)
            if query.lower() in json.dumps(inv["extracted_data"]).lower()
        ]
        choices = [f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}" for i, inv in filtered]

    return gr.update(choices=choices, value=None)


# Create Gradio Interface with Tabs
with gr.Blocks() as demo:

    gr.Markdown("# Invoice Processing System")

    with gr.Tabs() as tabs:
        # Extraction Tab
        with gr.Tab("Information Extraction", id=0):
            gr.Markdown("## Upload and Extract Invoice Information")

            file_upload = gr.File(
                label="Upload image files (Component)", file_count="multiple", type="filepath", file_types=["image"]
            )

            with gr.Row():
                extract_btn = gr.Button("Extract information", variant="secondary", scale=4)
                mark_all_btn = gr.Button("Mark all", scale=1)

            gr.Markdown("### Processed Invoices")

            # Dropdown to select invoice
            extraction_dropdown = gr.Dropdown(choices=[], label="Select invoice to view details", interactive=True)

            # Checkboxes for each invoice (max 10 for demo)
            checkboxes = []
            with gr.Column():
                for i in range(10):
                    with gr.Row(visible=False) as checkbox_row:
                        cb = gr.Checkbox(label=f"Invoice {i}", value=False)
                        checkboxes.append(cb)

            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                deny_btn = gr.Button("Deny", variant="secondary")

            # Modal for extraction view
            with gr.Column(visible=False) as extraction_modal:
                gr.Markdown("## Invoice Preview")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Image")
                        extraction_modal_image = gr.Image(label="", type="filepath")

                    with gr.Column():
                        gr.Markdown("### JSON Preview")
                        extraction_modal_json = gr.Code(label="", language="json", lines=20)

                close_extraction_modal_btn = gr.Button("Close", variant="secondary")

        # Approved Tab
        with gr.Tab("Approved Invoices", id=1):
            gr.Markdown("## Approved Invoices")

            with gr.Row():
                gr.Markdown("**Search**")
                search_box = gr.Textbox(placeholder="Hopkins", show_label=False, scale=4)

            # Dropdown for approved invoices
            approved_dropdown = gr.Dropdown(
                choices=[
                    f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
                    for i, inv in enumerate(processor.approved_invoices)
                ],
                label="Select invoice to view details",
                interactive=True,
            )

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

                close_approved_modal_btn = gr.Button("Close", variant="secondary")

    # Event handlers for Extraction Tab
    extract_btn.click(fn=extract_information, inputs=[file_upload], outputs=[extraction_dropdown]).then(
        fn=lambda: [
            (
                gr.update(visible=True, label=f"Approve {processor.processed_invoices[i]['file_name']}")
                if i < len(processor.processed_invoices)
                else gr.update(visible=False)
            )
            for i in range(10)
        ],
        inputs=[],
        outputs=checkboxes,
    )

    mark_all_btn.click(fn=mark_all, inputs=[], outputs=checkboxes)

    extraction_dropdown.change(
        fn=view_extraction_invoice,
        inputs=[extraction_dropdown],
        outputs=[extraction_modal_image, extraction_modal_json, extraction_modal],
    )

    approve_btn.click(
        fn=approve_invoices, inputs=checkboxes, outputs=[extraction_dropdown] + checkboxes + [approved_dropdown, tabs]
    )

    deny_btn.click(fn=deny_invoices, inputs=[], outputs=[extraction_dropdown] + checkboxes)

    # Event handlers for Approved Tab
    search_box.change(fn=search_approved, inputs=[search_box], outputs=[approved_dropdown])

    approved_dropdown.change(
        fn=view_approved_invoice,
        inputs=[approved_dropdown],
        outputs=[approved_modal_image, approved_modal_json, approved_modal],
    )

    close_extraction_modal_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[extraction_modal])

    close_approved_modal_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[approved_modal])

if __name__ == "__main__":
    demo.launch(
        share=False,
        # theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        """,
        # js=custom_js,
    )

# #  =========================================

# import asyncio
# import json
# import os
# import time
# from pathlib import Path
# from typing import Dict, List

# import gradio as gr

# # Path to store approved invoices
# APPROVED_INVOICES_FILE = "approved_invoices.json"


# class InvoiceProcessor:
#     def __init__(self):
#         self.processed_invoices = []
#         self.approved_invoices = self.load_approved_invoices()

#     def load_approved_invoices(self) -> List[Dict]:
#         """Load approved invoices from JSON file"""
#         if os.path.exists(APPROVED_INVOICES_FILE):
#             with open(APPROVED_INVOICES_FILE, "r") as f:
#                 return json.load(f)
#         return []

#     def save_approved_invoices(self):
#         """Save approved invoices to JSON file"""
#         with open(Path(__file__).parent / APPROVED_INVOICES_FILE, "w") as f:
#             json.dump(self.approved_invoices, f, indent=2)

#     async def extract_information_from_image(self, image_path: str, index: int) -> Dict:
#         """
#         Simulate VLM model extraction - Replace this with your actual VLM model
#         """
#         # Simulate processing time
#         await asyncio.sleep(1.5 + index * 0.3)

#         # Mock extracted data - replace with actual VLM inference
#         extracted_data = {
#             "invoice_nr": f"0909043{index}",
#             "date": "01/23/2021",
#             "seller": "Hopkins",
#             "client": "Sims PLC",
#             "total": f"${640.12 - index * 100:.2f}",
#             "items": [
#                 {"description": "Professional Services", "quantity": 1, "price": 350.00},
#                 {"description": "Consultation Fee", "quantity": 2, "price": 95.06},
#             ],
#             "address": "123 Main St, City, State",
#             "tax": "10%",
#         }

#         return {
#             "id": f"invoice_{int(time.time())}_{index}",
#             "file_path": image_path,
#             "file_name": Path(image_path).name,
#             "extracted_data": extracted_data,
#             "checked": False,
#         }


# # Initialize processor
# processor = InvoiceProcessor()


# def create_invoice_list_html():
#     """Create HTML list of invoices with checkboxes"""
#     if not processor.processed_invoices:
#         return "<div style='text-align: center; padding: 40px; color: #999;'>No invoices processed yet</div>"

#     html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"

#     for idx, invoice in enumerate(processor.processed_invoices):
#         checked = "checked" if invoice.get("checked", False) else ""

#         html += f"""
#         <div style='border: 1px solid #ddd; border-radius: 6px; padding: 16px; background: white;
#                     display: flex; align-items: center; gap: 16px;'>
#             <div style='flex: 1; display: flex; align-items: center; gap: 12px; cursor: pointer;'
#                  onclick='document.getElementById("invoice_selector").value = "{idx}";
#                           document.getElementById("invoice_selector").dispatchEvent(new Event("change"));'>
#                 <div style='padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px;
#                             background: #f5f5f5; min-width: 200px; text-align: center;'>
#                     {invoice['file_name']}
#                 </div>
#                 <span style='color: #999;'>→</span>
#             </div>
#             <div style='display: flex; flex-direction: column; align-items: center; gap: 4px;'>
#                 <input type='checkbox' {checked}
#                        onchange='fetch("/gradio_api/call/toggle_checkbox", {{
#                            method: "POST",
#                            headers: {{"Content-Type": "application/json"}},
#                            body: JSON.stringify({{data: [{idx}]}})
#                        }}).then(() => {{
#                            fetch("/gradio_api/call/refresh_list", {{
#                                method: "POST",
#                                headers: {{"Content-Type": "application/json"}},
#                                body: JSON.stringify({{data: []}})
#                            }}).then(r => r.json()).then(result => {{
#                                document.querySelector("#invoice_list iframe").contentDocument.body.innerHTML = result.data[0];
#                            }});
#                        }});'
#                        style='width: 18px; height: 18px; cursor: pointer;'>
#                 <span style='font-size: 11px; color: #666;'>Check box</span>
#             </div>
#         </div>
#         """

#     html += "</div>"
#     return html


# def create_approved_list_html(search_query: str = ""):
#     """Create HTML list of approved invoices"""
#     invoices = processor.approved_invoices

#     if search_query:
#         invoices = [inv for inv in invoices if search_query.lower() in json.dumps(inv["extracted_data"]).lower()]

#     if not invoices:
#         return "<div style='text-align: center; padding: 40px; color: #999;'>No approved invoices found</div>"

#     html = f"""
#     <div style='text-align: center; font-weight: 600; margin-bottom: 16px; font-size: 16px;'>
#         Listed Invoices ({len(invoices)})
#     </div>
#     <div style='display: flex; flex-direction: column; gap: 12px;'>
#     """

#     for idx, invoice in enumerate(invoices):
#         data = invoice["extracted_data"]

#         html += f"""
#         <div style='border: 1px solid #ddd; border-radius: 6px; padding: 20px; background: white;
#                     cursor: pointer; transition: background 0.2s;'
#              onmouseover='this.style.background="#f9fafb"'
#              onmouseout='this.style.background="white"'
#              onclick='document.getElementById("approved_selector").value = "{idx}";
#                       document.getElementById("approved_selector").dispatchEvent(new Event("change"));'>
#             <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px;'>
#                 <div>
#                     <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Invoice nr:</div>
#                     <div style='font-weight: 600;'>{data['invoice_nr']}</div>
#                 </div>
#                 <div>
#                     <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Date:</div>
#                     <div style='font-weight: 600;'>{data['date']}</div>
#                 </div>
#                 <div>
#                     <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Seller:</div>
#                     <div style='font-weight: 600;'>{data['seller']}</div>
#                 </div>
#                 <div>
#                     <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Client:</div>
#                     <div style='font-weight: 600;'>{data['client']}</div>
#                 </div>
#                 <div>
#                     <div style='font-size: 12px; color: #666; margin-bottom: 4px;'>Total:</div>
#                     <div style='font-weight: 600;'>{data['total']}</div>
#                 </div>
#             </div>
#         </div>
#         """

#     html += "</div>"
#     return html


# def extract_information(files):
#     """Extract information from uploaded files"""
#     if not files:
#         return create_invoice_list_html()

#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     processor.processed_invoices = []
#     for idx, file in enumerate(files):
#         result = loop.run_until_complete(processor.extract_information_from_image(file, idx))
#         processor.processed_invoices.append(result)

#     loop.close()
#     return create_invoice_list_html()


# def mark_all():
#     """Mark all invoices as checked"""
#     for invoice in processor.processed_invoices:
#         invoice["checked"] = True
#     return create_invoice_list_html()


# def toggle_checkbox(idx: int):
#     """Toggle checkbox state"""
#     if 0 <= idx < len(processor.processed_invoices):
#         processor.processed_invoices[idx]["checked"] = not processor.processed_invoices[idx].get("checked", False)
#     return create_invoice_list_html()


# def refresh_list():
#     """Refresh the invoice list"""
#     return create_invoice_list_html()


# def approve_invoices():
#     """Approve checked invoices and save to JSON file"""
#     checked_invoices = [inv for inv in processor.processed_invoices if inv.get("checked", False)]

#     if checked_invoices:
#         # Append to approved list
#         processor.approved_invoices.extend(checked_invoices)
#         processor.save_approved_invoices()

#         # Remove approved invoices from processed list
#         processor.processed_invoices = [inv for inv in processor.processed_invoices if not inv.get("checked", False)]

#         print(f"✓ Approved {len(checked_invoices)} invoice(s) and saved to {APPROVED_INVOICES_FILE}")

#     return (create_invoice_list_html(), create_approved_list_html(), gr.Tabs(selected=1))  # Switch to approved tab


# def deny_invoices():
#     """Clear all processed invoices"""
#     processor.processed_invoices = []
#     return create_invoice_list_html()


# def view_extraction_invoice(idx: str):
#     """View details for extraction invoice"""
#     try:
#         idx = int(idx)
#         if 0 <= idx < len(processor.processed_invoices):
#             invoice = processor.processed_invoices[idx]
#             return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
#     except:
#         pass
#     return None, "", gr.update(visible=False)


# def view_approved_invoice(idx: str):
#     """View details for approved invoice"""
#     try:
#         idx = int(idx)
#         if 0 <= idx < len(processor.approved_invoices):
#             invoice = processor.approved_invoices[idx]
#             return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
#     except:
#         pass
#     return None, "", gr.update(visible=False)


# def search_approved(query: str):
#     """Search approved invoices"""
#     return create_approved_list_html(query)


# # Create Gradio Interface
# with gr.Blocks() as demo:

#     gr.Markdown("# Invoice Processing System")

#     with gr.Tabs() as tabs:
#         # Extraction Tab
#         with gr.Tab("Information Extraction", id=0):
#             gr.Markdown("## Upload and Extract Invoice Information")

#             file_upload = gr.File(
#                 label="Upload image files (Component)", file_count="multiple", type="filepath", file_types=["image"]
#             )

#             with gr.Row():
#                 extract_btn = gr.Button("Extract information", variant="secondary", scale=4)
#                 mark_all_btn = gr.Button("Mark all", scale=1)

#             gr.Markdown("### Processed Invoices")

#             invoice_list = gr.HTML(value=create_invoice_list_html(), elem_id="invoice_list")

#             # Hidden selector for clicking invoices
#             invoice_selector = gr.Textbox(visible=False, elem_id="invoice_selector")

#             with gr.Row():
#                 approve_btn = gr.Button("Approve", variant="primary")
#                 deny_btn = gr.Button("Deny", variant="secondary")

#             # Modal for extraction view
#             with gr.Column(visible=False) as extraction_modal:
#                 gr.Markdown("## Invoice Preview")

#                 with gr.Row():
#                     with gr.Column():
#                         gr.Markdown("### Image")
#                         extraction_modal_image = gr.Image(label="", type="filepath")

#                     with gr.Column():
#                         gr.Markdown("### JSON Preview")
#                         extraction_modal_json = gr.Code(label="", language="json", lines=20)

#                 close_extraction_modal_btn = gr.Button("Close", variant="secondary")

#         # Approved Tab
#         with gr.Tab("Approved Invoices", id=1):
#             gr.Markdown("## Approved Invoices")

#             with gr.Row():
#                 gr.Markdown("**Search**")
#                 search_box = gr.Textbox(placeholder="Hopkins", show_label=False, scale=4)

#             approved_list = gr.HTML(value=create_approved_list_html(), elem_id="approved_list")

#             # Hidden selector for clicking approved invoices
#             approved_selector = gr.Textbox(visible=False, elem_id="approved_selector")

#             # Modal for approved view
#             with gr.Column(visible=False) as approved_modal:
#                 gr.Markdown("## Invoice Preview")

#                 with gr.Row():
#                     with gr.Column():
#                         gr.Markdown("### Image")
#                         approved_modal_image = gr.Image(label="", type="filepath")

#                     with gr.Column():
#                         gr.Markdown("### JSON Preview")
#                         approved_modal_json = gr.Code(label="", language="json", lines=20)

#                 close_approved_modal_btn = gr.Button("Close", variant="secondary")

#     # Event handlers for Extraction Tab
#     extract_btn.click(fn=extract_information, inputs=[file_upload], outputs=[invoice_list])

#     mark_all_btn.click(fn=mark_all, inputs=[], outputs=[invoice_list])

#     invoice_selector.change(
#         fn=view_extraction_invoice,
#         inputs=[invoice_selector],
#         outputs=[extraction_modal_image, extraction_modal_json, extraction_modal],
#     )

#     approve_btn.click(fn=approve_invoices, inputs=[], outputs=[invoice_list, approved_list, tabs])

#     deny_btn.click(fn=deny_invoices, inputs=[], outputs=[invoice_list])

#     close_extraction_modal_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[extraction_modal])

#     # Event handlers for Approved Tab
#     search_box.change(fn=search_approved, inputs=[search_box], outputs=[approved_list])

#     approved_selector.change(
#         fn=view_approved_invoice,
#         inputs=[approved_selector],
#         outputs=[approved_modal_image, approved_modal_json, approved_modal],
#     )

#     close_approved_modal_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[approved_modal])

#     # API endpoints for checkbox toggling
#     toggle_checkbox_api = gr.Number(visible=False)
#     toggle_checkbox_api.change(
#         fn=toggle_checkbox, inputs=[toggle_checkbox_api], outputs=[invoice_list], api_name="toggle_checkbox"
#     )

#     refresh_list_api = gr.Button(visible=False)
#     refresh_list_api.click(fn=refresh_list, inputs=[], outputs=[invoice_list], api_name="refresh_list")


# if __name__ == "__main__":
#     demo.launch(
#         share=False,
#         # theme=gr.themes.Soft(),
#         css="""
#         .container { max-width: 1200px; margin: auto; }
#     """,
#     )
