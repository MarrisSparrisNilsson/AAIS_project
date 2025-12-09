# import asyncio
# import base64
# import io
# import json
# import os
# import time
# from pathlib import Path
# from typing import Dict, List, Tuple

# import gradio as gr
# from PIL import Image

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
#         with open(APPROVED_INVOICES_FILE, "w") as f:
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


# def create_invoice_list_html(invoices: List[Dict], show_checkboxes: bool = True) -> str:
#     """Create HTML for invoice list"""
#     if not invoices:
#         return "<div style='text-align: center; padding: 40px; color: #666;'>No invoices to display</div>"

#     html = "<div style='display: flex; flex-direction: column; gap: 16px;'>"

#     for idx, invoice in enumerate(invoices):
#         checked_attr = "checked" if invoice.get("checked", False) else ""
#         checkbox_html = (
#             f"""
#             <div style='display: flex; flex-direction: column; align-items: center; gap: 4px;'>
#                 <input type='checkbox' id='checkbox_{idx}' {checked_attr}
#                        style='width: 20px; height: 20px; cursor: pointer;'
#                        onchange='document.getElementById("checkbox_state_{idx}").value = this.checked'>
#                 <span style='font-size: 12px; color: #666;'>Check box</span>
#             </div>
#         """
#             if show_checkboxes
#             else ""
#         )

#         html += f"""
#         <div style='border: 1px solid #ddd; border-radius: 8px; padding: 16px;
#                     display: flex; align-items: center; gap: 16px; background: white;
#                     transition: background 0.2s;'
#              onmouseover='this.style.background="#f9fafb"'
#              onmouseout='this.style.background="white"'>

#             <div style='width: 80px; height: 80px; background: #e5e7eb;
#                         border-radius: 4px; display: flex; align-items: center;
#                         justify-content: center; font-size: 10px; color: #666; text-align: center;'>
#                 Invoice<br>image<br>preview
#             </div>

#             <div style='flex: 1; padding: 8px;'>
#                 <div style='font-weight: 500;'>{invoice['file_name']}</div>
#                 <div style='font-size: 12px; color: #666; margin-top: 4px;'>
#                     Invoice: {invoice['extracted_data']['invoice_nr']} |
#                     Total: {invoice['extracted_data']['total']}
#                 </div>
#             </div>

#             <button onclick='window.open_invoice_modal({idx})'
#                     style='padding: 8px 16px; background: white; border: 1px solid #ddd;
#                            border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 4px;'
#                     onmouseover='this.style.background="#f3f4f6"'
#                     onmouseout='this.style.background="white"'>
#                 View Details →
#             </button>

#             {checkbox_html}
#         </div>
#         """

#     html += "</div>"
#     return html


# def create_approved_list_html(invoices: List[Dict], search_query: str = "") -> str:
#     """Create HTML for approved invoices list"""
#     if search_query:
#         invoices = [inv for inv in invoices if search_query.lower() in json.dumps(inv["extracted_data"]).lower()]

#     if not invoices:
#         return "<div style='text-align: center; padding: 40px; color: #666;'>No approved invoices found</div>"

#     html = f"""
#     <div style='text-align: center; font-weight: 600; margin-bottom: 16px; font-size: 16px;'>
#         Listed Invoices ({len(invoices)})
#     </div>
#     <div style='display: flex; flex-direction: column; gap: 16px;'>
#     """

#     for idx, invoice in enumerate(invoices):
#         data = invoice["extracted_data"]
#         html += f"""
#         <div style='border: 1px solid #ddd; border-radius: 8px; padding: 20px;
#                     background: white; cursor: pointer; transition: background 0.2s;'
#              onclick='window.open_invoice_modal({idx})'
#              onmouseover='this.style.background="#f9fafb"'
#              onmouseout='this.style.background="white"'>

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


# async def process_images_async(files: List[str]):
#     """Process images asynchronously and update the list progressively"""
#     if not files:
#         yield create_invoice_list_html([])
#         return

#     processor.processed_invoices = []

#     # Process each image
#     tasks = [processor.extract_information_from_image(file, idx) for idx, file in enumerate(files)]

#     for task in asyncio.as_completed(tasks):
#         result = await task
#         processor.processed_invoices.append(result)
#         # In a real implementation, you'd use gr.update() with streaming
#         yield create_invoice_list_html(processor.processed_invoices)

#     # return create_invoice_list_html(processor.processed_invoices)


# def extract_information(files):
#     """Wrapper for async processing"""
#     if not files:
#         return create_invoice_list_html([])

#     # Run async processing
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     processor.processed_invoices = []
#     for idx, file in enumerate(files):
#         result = loop.run_until_complete(processor.extract_information_from_image(file, idx))
#         processor.processed_invoices.append(result)

#     loop.close()
#     return create_invoice_list_html(processor.processed_invoices)


# def mark_all():
#     """Mark all invoices as checked"""
#     for invoice in processor.processed_invoices:
#         invoice["checked"] = True
#     return create_invoice_list_html(processor.processed_invoices)


# def toggle_checkbox(invoice_idx: int):
#     """Toggle checkbox state for an invoice"""
#     if 0 <= invoice_idx < len(processor.processed_invoices):
#         processor.processed_invoices[invoice_idx]["checked"] = not processor.processed_invoices[invoice_idx]["checked"]
#     return create_invoice_list_html(processor.processed_invoices)


# def approve_invoices():
#     """Approve checked invoices and save to file"""
#     checked_invoices = [inv for inv in processor.processed_invoices if inv.get("checked", False)]

#     if checked_invoices:
#         processor.approved_invoices.extend(checked_invoices)
#         processor.save_approved_invoices()
#         processor.processed_invoices = [inv for inv in processor.processed_invoices if not inv.get("checked", False)]

#     return (
#         create_invoice_list_html(processor.processed_invoices),
#         create_approved_list_html(processor.approved_invoices),
#         gr.update(visible=False),  # Hide extraction view
#         gr.update(visible=True),  # Show approved view
#     )


# def deny_invoices():
#     """Clear all invoices"""
#     processor.processed_invoices = []
#     return create_invoice_list_html([])


# def switch_to_extraction():
#     """Switch to extraction view"""
#     return (
#         gr.update(visible=True),  # Show extraction view
#         gr.update(visible=False),  # Hide approved view
#         create_invoice_list_html(processor.processed_invoices),
#     )


# def switch_to_approved():
#     """Switch to approved view"""
#     return (
#         gr.update(visible=False),  # Hide extraction view
#         gr.update(visible=True),  # Show approved view
#         create_approved_list_html(processor.approved_invoices),
#     )


# def search_approved(query: str):
#     """Search approved invoices"""
#     return create_approved_list_html(processor.approved_invoices, query)


# def get_invoice_details(invoice_idx: int, from_approved: bool = False):
#     """Get invoice details for modal"""
#     invoices = processor.approved_invoices if from_approved else processor.processed_invoices

#     if 0 <= invoice_idx < len(invoices):
#         invoice = invoices[invoice_idx]
#         return (invoice["file_path"], json.dumps(invoice["extracted_data"], indent=2), gr.update(visible=True))
#     return None, "", gr.update(visible=False)


# # Create Gradio Interface
# with gr.Blocks() as demo:

#     gr.Markdown("# Invoice Processing System")

#     # Navigation buttons
#     with gr.Row():
#         extraction_nav_btn = gr.Button("Information Extraction", variant="primary", elem_classes=["nav-button"])
#         approved_nav_btn = gr.Button(
#             f"Approved Invoices ({len(processor.approved_invoices)})", elem_classes=["nav-button"]
#         )

#     # Extraction View
#     with gr.Column(visible=True) as extraction_view:
#         gr.Markdown("## Upload and Extract Invoice Information")

#         file_upload = gr.File(
#             label="Upload image files (Component)", file_count="multiple", type="filepath", file_types=["image"]
#         )

#         with gr.Row():
#             extract_btn = gr.Button("Extract information", variant="secondary", scale=4)
#             mark_all_btn = gr.Button("Mark all", scale=1)

#         invoice_list = gr.HTML(value=create_invoice_list_html([]), label="Processed Invoices")

#         with gr.Row():
#             approve_btn = gr.Button("Approve", variant="primary")
#             deny_btn = gr.Button("Deny", variant="secondary")

#     # Approved View
#     with gr.Column(visible=False) as approved_view:
#         gr.Markdown("## Approved Invoices")

#         with gr.Row():
#             gr.Markdown("**Search**")
#             search_box = gr.Textbox(placeholder="Hopkins", show_label=False, scale=4)

#         approved_list = gr.HTML(value=create_approved_list_html(processor.approved_invoices), label="Approved Invoices")

#     # Modal for invoice preview
#     with gr.Column(visible=False) as modal:
#         gr.Markdown("## Invoice Preview")

#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("### Image")
#                 modal_image = gr.Image(label="", type="filepath")

#             with gr.Column():
#                 gr.Markdown("### JSON Preview")
#                 modal_json = gr.Code(label="", language="json", lines=20)

#         close_modal_btn = gr.Button("Close", variant="secondary")

#     # Event handlers
#     extract_btn.click(fn=extract_information, inputs=[file_upload], outputs=[invoice_list])

#     mark_all_btn.click(fn=mark_all, inputs=[], outputs=[invoice_list])

#     approve_btn.click(
#         fn=approve_invoices, inputs=[], outputs=[invoice_list, approved_list, extraction_view, approved_view]
#     )

#     deny_btn.click(fn=deny_invoices, inputs=[], outputs=[invoice_list])

#     extraction_nav_btn.click(fn=switch_to_extraction, inputs=[], outputs=[extraction_view, approved_view, invoice_list])

#     approved_nav_btn.click(fn=switch_to_approved, inputs=[], outputs=[extraction_view, approved_view, approved_list])

#     search_box.change(fn=search_approved, inputs=[search_box], outputs=[approved_list])

#     close_modal_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[modal])

# if __name__ == "__main__":
#     demo.launch(
#         share=False,
#         # theme=gr.themes.Soft(),
#         css="""
#             .container { max-width: 1200px; margin: auto; }
#             .nav-button { min-width: 200px; }
#         """,
#     )


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
        with open(APPROVED_INVOICES_FILE, "w") as f:
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
    """Approve checked invoices and save to file"""
    # Update checkbox states
    for i, checked in enumerate(checkbox_values[: len(processor.processed_invoices)]):
        if i < len(processor.processed_invoices):
            processor.processed_invoices[i]["checked"] = checked

    checked_invoices = [inv for inv in processor.processed_invoices if inv.get("checked", False)]

    if checked_invoices:
        processor.approved_invoices.extend(checked_invoices)
        processor.save_approved_invoices()
        processor.processed_invoices = [inv for inv in processor.processed_invoices if not inv.get("checked", False)]

    # Update extraction view
    choices = [
        f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
        for i, inv in enumerate(processor.processed_invoices)
    ]

    # Update approved view
    approved_choices = [
        f"{i}: {inv['file_name']} - {inv['extracted_data']['invoice_nr']}"
        for i, inv in enumerate(processor.approved_invoices)
    ]

    # Return: extraction_dropdown + 10 checkboxes + approved_dropdown + tabs
    checkbox_updates = get_checkboxes_state()
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
