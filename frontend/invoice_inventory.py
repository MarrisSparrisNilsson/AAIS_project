from pathlib import Path

import gradio as gr
import pandas as pd

# --- Dummy data (replace with your own structured data) ---
data = [
    {
        "invoice": "09090431",
        "date": "01/23/2021",
        "seller": "Hopkins",
        "client": "Sims PLC",
        "total": 640.12,
        "png": Path("invoices_dataset/unified_dataset/images/dataset1_katanaml_test_katanaml_0001.png").resolve(),
    },
    {
        "invoice": "09090432",
        "date": "01/24/2021",
        "seller": "Hopkins",
        "client": "Sims PLC",
        "total": 540.12,
        "png": Path("invoices_dataset/unified_dataset/images/dataset1_katanaml_test_katanaml_0002.png").resolve(),
    },
    {
        "invoice": "09090433",
        "date": "01/25/2021",
        "seller": "Hopkins",
        "client": "Sims PLC",
        "total": 440.12,
        "png": Path("invoices_dataset/unified_dataset/images/dataset1_katanaml_test_katanaml_0003.png").resolve(),
    },
    {
        "invoice": "09090434",
        "date": "01/26/2021",
        "seller": "Hopkins",
        "client": "Sims PLC",
        "total": 340.12,
        "png": Path("invoices_dataset/unified_dataset/images/dataset1_katanaml_test_katanaml_0004.png").resolve(),
    },
    {
        "invoice": "09090435",
        "date": "01/27/2021",
        "seller": "Hopkins",
        "client": "Sims PLC",
        "total": 240.12,
        "png": Path("invoices_dataset/unified_dataset/images/dataset1_katanaml_test_katanaml_0005.png").resolve(),
    },
]

df = pd.DataFrame(data)


# --- Logic: filter invoices ---
def filter_invoices(query):
    if not query:
        filtered = df
    else:
        filtered = df[
            df.apply(
                lambda row: query.lower() in str(row["invoice"]).lower()
                or query.lower() in str(row["date"]).lower()
                or query.lower() in str(row["seller"]).lower()
                or query.lower() in str(row["client"]).lower()
                or query.lower() in str(row["total"]).lower(),
                axis=1,
            )
        ]

    count_element = gr.Markdown(f"## Listed Invoices ({len(filtered)})")

    cards_html = ""
    button_ids = []

    for idx, row in filtered.iterrows():

        btn_id = f"btn_{idx}"
        png_path = row["png"]  # the dataframe must contain a 'png' column with paths to images

        button_ids.append((btn_id, png_path))

        cards_html += f"""
        <div onclick="document.getElementById('{btn_id}').click();"
            class="invoice-card"
            style='border:1px solid #aaa; padding:15px; margin:10px 0; border-radius:8px; display:flex; justify-content:end; align-items:center;'>
                <div style='display:flex; justify-content:space-between; align-items:center; width:100%'>
                    <div class="invoice-card-info"><b>Invoice nr:</b> {row['invoice']}</div>
                    <div class="invoice-card-info"><b>Date:</b> {row['date']}</div>
                    <div class="invoice-card-info"><b>Seller:</b> {row['seller']}</div>
                    <div class="invoice-card-info"><b>Client:</b> {row['client']}</div>
                    <div class="invoice-card-info"><b>Total:</b> ${row['total']:.2f}</div>
                </div>
                <div style='font-size:30px; color:#666; padding: 0 0 0 20px; height: 50px'>&#8250;</div>
        </div>
        """

    return count_element, cards_html, button_ids


def show_png(png_path):
    return gr.HTML(f"<iframe src='{png_path}' width='100%' height='800px'></iframe>")


def invoice_inventory():
    with gr.Blocks() as demo:

        # Inject CSS (new Gradio method)
        with open(Path("frontend/style.css").resolve()) as f:
            css = f.read()
        gr.HTML(f"<style>{css}</style>")

        # Begin UI layout
        gr.Markdown("# Invoice Browser")

        with gr.Column():
            gr.HTML(
                """
                        <h3>
                            Search and browse invoices. Click on an invoice card to view the full invoice.
                        </h3>
                    """
            )
            search_input = gr.Textbox(placeholder="Hopkins", scale=2, label="Search Invoices")

        invoice_count, initial_cards, _ = filter_invoices("")
        cards_output = gr.HTML(initial_cards)

        hidden_buttons_area = gr.Column(visible=True)

        png_viewer = gr.HTML("")  # area where png is shown

        def update_cards(query):
            count_text, cards_html, button_info = filter_invoices(query)

            # clear hidden button area and rebuild it
            hidden_buttons_area.children = []
            for btn_id, png_path in button_info:
                b = gr.Button("", elem_id=btn_id)
                b.click(show_png, inputs=gr.State(png_path), outputs=png_viewer)

            return count_text, cards_html

        search_input.change(update_cards, inputs=search_input, outputs=[invoice_count, cards_output])

        return demo
        # search_input.change(filter_invoices, inputs=search_input, outputs=[invoice_count, cards_output])


if __name__ == "__main__":

    if False:
        print(Path("invoices_dataset/unified_dataset/images/dataset1_katanaml_test_katanaml_0001.png").resolve())
        # print("CSS Path:", Path("frontend/style.css").resolve())
    else:

        demo = invoice_inventory()

        demo.launch()
