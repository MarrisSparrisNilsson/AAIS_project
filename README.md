# Invoice Information Extractor

### Team members:

|      Name       |    GitHub Handle     |
| :-------------: | :------------------: |
|   Aron Kesete   |        AronKG        |
| Isac Gustafsson |       Aaetpio        |
| Martin Nilsson  | MarrisSparrisNilsson |

## Description

### Motivation

<!-- What is the real-world problem being tackled? -->

In a real world scenario, businesses handle a lot of invoice documents that are to be processed and information to be extracted from them. Some businesses might already have an automatic document process pipeline that is triggered right when an order is placed and stores all information in a database. On the other hand, if documents are not automatically handled, this project aims to handle invoices by scanning these documents with the help of state-of-the-art Optical Character Recognition (OCR) and extracting relevant information (such as invoice number, invoice date, total cost).

### Pre-trained model/method

<!-- What pre-trained AI/ML models or algorithms are planned to be used and improved. The project can focus on issues other than accuracy (e.g., time, memory, parallelization etc.) -->

To get an understanding of what models might be of most use to us, we found [this survey](https://ieeexplore.ieee.org/document/11193825) by Khan et al., published on October 6th, 2025, which details the prominence of various machine learning methods for the task of text extraction. According to its findings, Visual Transformers (ViT) and Generative Adversarial Networks (GAN) are the most prominent architectures today, with ViTs being better for multilingual documents and GANs being better when the quality of the image is poor. Given that our current datasets mostly consist of clear images and PDFs, with a few different languages present, we draw the conclusion that focusing on ViTs is most appropriate.

According to our estimations, multi-modal VLMs with a maximum of 6B billion parameters or less would be suited for our application in order to run it on our local machine. If the accuracy becomes too poor or we find that the capabilities of the model is too limited, we will look at models with a larger amount of parameters and try to run it on an A-100 GPU.

Reviewing the models available on [Hugging Face](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=downloads) revealed that when sorting on most downloaded models for the task of Image-text-to-Text, four out of the top five are versions of the Qwen-model.

Given the above findings, we elected to utilize [**Qwen3-VL-2B**](https://huggingface.co/docs/transformers/model_doc/qwen3_vl), which is a multi-modal vision-language model that is good for visual understanding and processing of text information. Our plan is to fine-tune this model for the tasks outlined below.

#### Planned program flow:
![Invoice -> Model -> Structured output of Invoice -> Model (OCR) -> Text -> Model (Structure of important info) -> Structured output -> Enter invoice information in database -> Display in inventory UI.](README_images/idea.jpg)

1. **Input:** User provides an invoice as an image or PDF.
2. **OCR:** The invoice is passed to an OCR Engine (most likely implemented via a ViT) which solely converts the text on the image into a text file.
3. **Extraction:** The generated text file is passed to an agent fine-tuned to extract particular pieces of information, such as _invoice number_, _products purchased_, _total cost_, and outputs this in a structured JSON document.
4. **Control:** The structured JSON document is passed, along with the original invoice, to both a VLM and a human critic (the user) for comparison. The purpose of the VLM is to act as a second critic, who passes its conclusion (along with the confidence of said conclusion) to the human critic. The human critic will be shown the original invoice and the generated JSON document (most likely presented in a more readable format than pure JSON) side-by-side, along with the VLMs assessment and confidence. The human critic can either approve or reject; if approved, the system continues to step 5 and if rejected, the system could re-run steps 2-4 or abort the process, allowing the user to start over from the beginning.
5. **Query Generation & Execution:** The JSON document is passed to an agent fine-tuned to generate SQL queries to add or update the extracted information to the database, and these queries are autonomously executed through a Model Context Protocol (MCP) connection to the database.
6. **Visible Change:** The user can inspect the updated database directly through a web-based UI.

#### AI Component: System Overview
- User Interaction:
    End-users interact with the system through a web-based UI. They upload an invoice image file (e.g., PNG or JPG format) via a drag-and-drop interface or file selector. The system then processes the invoice automatically and displays the extracted information for review.
- Outputs:
    The system generates structured JSON documents containing the extracted invoice information, such as invoice number, products purchased, total cost, and vendor details. 

### Experiment and Dataset

<!-- What dataset is planned to be used, and how to collect data for the project -->

The primary datasets that will be used are various invoice datasets gathered from _Huggingface_. The datasets include **images of invoices** (currently +2000 images) together with the **truth text data** within the images in json format.

**Links to datasets:**

-   https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1
-   https://huggingface.co/datasets/doceoSoftware/docvqa_invoices_v1
-   https://huggingface.co/datasets/Aoschu/German_invoices_dataset
-   https://huggingface.co/datasets/michalaerson/annotated-energy-invoices
-   https://huggingface.co/datasets/ilhamxx/xdata_invoices
-   https://huggingface.co/datasets/featsystems/invoices

---

**Example invoices from datasets:**

<img src="README_images/dataset1_katanaml_train_katanaml_0109.jpg" alt="Example 1" width=450/>

<img src="README_images/dataset2_docvqa_train_docvqa_1511.png" alt="Example 2" width=450/>

---

**Experiment Details:**

Our first fine-tuning experiments consisted of tuning the OCR model to solely output the detected invoice number, date of issue, and total gross amount. These experiments are implemented in the `fine_tune_mlflow.ipynb` notebook along with the utilized hyperparameters, optimizers and metrics. To cut down on the training time, we only fine-tune about 1.1% of the model's 2 billion parameters. Below are a few screenshots from MLFlow:

<img src="README_images/runs.png" alt="Screenshot from MLFlow of Runs screen" width=650/>

This screenshot shows the runs performed using the latest model.

<img src="README_images/loss.png" alt="Screenshot from MLFlow of loss metric decreasing over the training steps" width=650/>

This screenshot shows the loss metric decreasing over the training steps as the model learns to adhere to the requested output, indicating that the fine-tuning process is successful.

<img src="README_images/artifacts.png" alt="Screenshot from MLFlow of artifact screen of last succesful run" width=650/>

This screenshot shows the artifacts produced by our latest run. Among the artifacts are the dependencies of the model (in terms of Python packages) and the checkpoint for the trained parts of the model at the final training step.

**Experiment Insights**

Based on the evaluation that we have performed (on 200 images from the first dataset), our limited attempts at fine-tuning have not improved the model's performance. As can be seen in the runs logged to MLFlow, the base, pre-trained model outperforms the fine-tuned version considerably, with an overall accuracy of around 87% (0.87), compared to the fine-tuned's 24% (0.24). This result likely stems from the extremely limited dataset cleaned and prepared for the fine-tuning process, and we believe that by incorporating more of the datasets that we selected, we could substantially improve the results.

## Project Installation and Setup

If you are using **Conda environments**, run:

```bash
conda env create -f environment.yml
```

If you are using **virtual Python environments**:

> Run `install_dependencies.py` to install `requirements.txt` packages and `pytorch` installation.

Or do it manually by running:

```bash
py -m pip install -r requirements.txt
```

---

If you want to start the MLFlow server locally (and thus be able to observe the results of the logged runs), navigate to the mlflow directory under src:

```bash
cd src/mlflow
```
... then run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

To run an initial test of the training and fine tuning with 3 images from the training set using the Qwen3-VL model, follow the instructions inside the `fine_tune_mlflow.ipynb` notebook.

**Note:** Running the fine-tuning and evaluation requires access to a **cuda GPU** with around 8-12 GB VRAM.

To run a simple evaluation script comparing the base, pre-trained model to the latest fine-tuned model, navigate to the mlflow directory under src:
```bash
cd src/mlflow
```
... then run:
```bash
python evaluation.py
```

---

Run docker compose to start up services:

```bash
docker-compose up -d --build
```
**--build** should be skipped on subsequent runs, only required to build the images for the first run (or after changes have been made to either code or model)

Docker down or shutdown services:

```bash
docker-compose down
```

**Note:** Starting the inference service and loading the model requires access to a **cuda GPU** with around 8-12 GB VRAM, which must also be made accesible to the Docker containers through installation of the NVIDA Container Toolkit ([instructions found here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) on the host machine.

**Additional Note:** The total size of the created images sits around 16-17 GB, with the inference server taking up 14-15 of those. Keep this in mind when building the images.

### Update `requirements.txt`

> Run the `update_requirements.py` (found under src/utility) to generate _top-level dependencies_ from your current `.venv`.

## Deployment

For upcoming deployment steps we will be using Docker with docker compose files to start up services, such as the `invoice-api` and `gradio-ui`.

**MLFlow** can be used to serve the latest pre-trained model to docker.

**FastAPI** will act as the hosting server which provides the API for our application that is well suited for production.

### Model Deployment and Inference Serving

The system is deployed locally using Docker Compose with two main services:

1. **Invoice Processing API** (FastAPI) - Port 8000
   - Loads the fine-tuned Qwen3-VL model from MLFlow checkpoints
   - Provides REST API for invoice processing
   - Model checkpoints are stored in `./src/mlflow/mlruns/`

2. **Web Interface** (Gradio UI) - Port 7861
   - User-friendly interface for uploading invoices
   - Displays extracted information
   - Communicates with the API backend

**How to Use:**

After starting with `docker-compose up`:

**Option 1: Use the Web Interface (Recommended)**
- Go to: http://localhost:7861
- Upload an invoice image
- View extracted information directly in the browser

**Option 2: Use the API Directly**
```bash
curl -X POST "http://localhost:8000/process-invoice" \
  -F "file=@path/to/invoice.jpg"
```


## Progressive Design Updates

We decided to change the used model to an alternative Qwen3-VL model named: [Qwen3-VL-2B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit). We noticed that the previous model took a long time to run locally and therefore sought out a more lightweight version. This one utilizes quantization, which essentially means that the precision of the data type is decreased to save memory and computation.

#### Actual program flow:
![Invoice -> Model -> Structured output of Invoice -> Model (OCR) -> Text -> Model (Structure of important info) -> Structured output -> Enter invoice information in database -> Display in inventory UI.](README_images/c_idea.png)

## Code/Docker file references
**Repository Structure:**
- `docker-compose.yml` - Docker setup for all services
- `src/mlflow/fine_tune_mlflow.ipynb` - Main training notebook
- `src/model/app/` - FastAPI application code
- `requirements.txt` & `environment.yml` - Dependencies


## Conclusion and Reflection

We attempted to implement an invoice processing system using a pre-trained Qwen3-VL-2B model. Docker containerization worked well for consistent deployment, and MLFlow effectively tracked experiments. FastAPI provided a robust API for the application.

However, the main challenge was fine-tuning, where results sometimes worsened rather than improved, and integrating all components into a complete pipeline required significant effort.

Given more time, we would focus on improving the project structure, conducting more thorough testing, and experimenting with better fine-tuning approaches.

The current solution remains a development prototype with limitations in handling diverse invoice formats and lacks production features like load balancing or advanced monitoring.
