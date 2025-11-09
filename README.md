# Invoice Information Extractor

## Description

### Motivation

<!-- What is the real-world problem being tackled? -->

In a real world scenario, businesses handle a lot of invoice documents that are to be processed and information to be extracted from them. Some businesses might already have an automatic document process pipeline that is triggered right when an order is placed and stores all information in a database. On the other hand, if documents are not automatically handled, this project aims to handle invoices by scanning these documents with the help of state-of-the-art Optical Character Recognition (OCR) regardless of document structure but it should still be a categorized as an invoice.

### Pre-trained model/method

<!-- What pre-trained AI/ML models or algorithms are planned to be used and improved. The project can focus on issues other than accuracy (e.g., time, memory, parallelization etc.) -->

According to our estimations, multi-modal VLMs with a maximum of 6B billion parameters or less would be suited for our application in order to run it on our local machine. If it becomes necessary we will look at models with a larger amount of parameters and try to run it on an A-100 GPU.

As an initial model test, we will experiment with the [**Qwen3-VL-2B**](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) which is a multi-modal vision-language model that is good for visual understanding and processing of text information.

**Planned program flow:**

Invoice $\rightarrow$ Model $\rightarrow$ Structured output of Invoice $\rightarrow$ Model (OCR) $\rightarrow$ Text $\rightarrow$ Model (Structure of important info) $\rightarrow$ Structured output $\rightarrow$ Enter invoice information in database $\rightarrow$ Display in inventory UI.

### Dataset

<!-- What dataset is planned to be used, and how to collect data for the project -->

The datasets that will be used are various invoice datasets gathered from sources such as _Huggingface_ or _Kaggle_. The dataset include **images of invoices** (currently +2000 images) together with the **truth text data** within the images in json format.

---

### Team members:

-   Aron Kesete (AronKG)
-   Isac Gustafsson (Aaetpio)
-   Martin Nilsson (MarrisSparrisNilsson)
