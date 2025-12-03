# %%
import torch
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit", load_in_4bit=True, use_gradient_checkpointing="unsloth"
)

# %%
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# %%
from datasets import load_dataset

dataset = load_dataset("invoices_dataset/mini_dataset/images", split="train")

# %%
instruction = "Read the OCR in the image."


def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}, {"type": "image", "image": sample["image"]}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["invoice_nr"]}]},
    ]
    return {"messages": conversation}


pass

# %%
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# %%
FastVisionModel.for_inference(model)

image = dataset[2]["image"]
instruction = "Read the OCR in the image."

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=400, use_cache=True, temperature=1.5, min_p=0.1)

import os

# %%
import mlflow

cwd = os.getcwd()
if os.access(cwd, os.W_OK):
    db_path = os.path.abspath("mlflow.db")
else:
    home_mlflow_dir = os.path.expanduser("~/mlflow_local")
    os.makedirs(home_mlflow_dir, exist_ok=True)
    db_path = os.path.join(home_mlflow_dir, "mlflow.db")

os.makedirs(os.path.dirname(db_path), exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("qwen3-vl-invoice-finetune")

from datetime import datetime

from trl import SFTConfig, SFTTrainer

# %%
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="mlflow",
        run_name=f"qwen3-vl-invoice-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        # Below items are required for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

# %%
trainer.train()

# %%
FastVisionModel.for_inference(model)

image = dataset[2]["image"]
instruction = "Read the OCR in the image."

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

# %%
last_run_id = mlflow.last_active_run().info.run_id

with mlflow.start_run(run_id=last_run_id):
    mlflow.log_param("model_name", "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit")
    mlflow.log_param("finetune_task", "invoice_number_extraction")
    mlflow.log_params(model.peft_config)

    # Path where SFTTrainer saved the checkpoint/adapter
    adapter_path = "outputs/checkpoint-30"
    assert os.path.isdir(adapter_path), f"Adapter folder not found: {adapter_path}"

    import mlflow.pyfunc

    class PEFTVisionWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            adapter_local = context.artifacts["adapter"]

            from peft import PeftModel
            from unsloth import FastVisionModel

            # Load base model & tokenizer
            base_name = "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit"
            model, tokenizer = FastVisionModel.from_pretrained(
                base_name,
                load_in_4bit=True,
                use_gradient_checkpointing="unsloth",
            )

            # Attach the saved PEFT adapters
            model = PeftModel.from_pretrained(model, adapter_local)

            self.model = model.eval()
            self.tokenizer = tokenizer

        def predict(self, context, model_input):
            import torch
            from PIL import Image

            results = []
            for _, row in model_input.iterrows():
                img = row["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                instruction = row.get("instruction", "Read the OCR in the image.")

                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction}],
                    }
                ]
                input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.tokenizer(img, input_text, return_tensors="pt").to(next(self.model.parameters()).device)
                with torch.no_grad():
                    gen = self.model.generate(**inputs, max_new_tokens=400)
                decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
                results.append(decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded))
            import pandas as pd

            return pd.DataFrame({"prediction": results})

    mlflow.pyfunc.log_model(
        python_model=PEFTVisionWrapper(), name="qwen3vl_finetuned_extraction", artifacts={"adapter": adapter_path}
    )
