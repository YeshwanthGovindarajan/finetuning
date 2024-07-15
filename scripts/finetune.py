
```python
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

project_name = 'finetune-llm'
model_name = 'gpt2'

push_to_hub = True
hf_token = os.getenv("HF_TOKEN")
repo_id = "repoid"

learning_rate = 2e-4
num_epochs = 1
batch_size = 1
block_size = 1024
warmup_ratio = 0.1
weight_decay = 0.01
gradient_accumulation = 4
mixed_precision = "fp16"
peft = True
quantization = "int4"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

data_path = "data/user_twit_bios.csv"
dataset = load_dataset('csv', data_files=data_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

if peft:
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=['q', 'v'],  # This is specific to T5; adjust as necessary for other models
        dtype=torch.float16
    )
    model = get_peft_model(model, config)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=block_size)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("text", "input_ids")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    logging_dir='./logs',
    push_to_hub=push_to_hub,
    hub_model_id=repo_id,
    hub_token=hf_token,
    fp16=(mixed_precision == "fp16")
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()

# Push to Hub
if push_to_hub:
    trainer.push_to_hub()
