import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# Configuration
project_name = 'finetune-llm'
model_name = 'gpt2'
save_model_path = "./finetuned_model"
hf_token = os.getenv("HF_TOKEN")
data_path = "data/user_twit_bios.csv"

learning_rate = 2e-4
num_epochs = 1
batch_size = 1
block_size = 1024
peft = True
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Load Dataset
dataset = load_dataset('csv', data_files=data_path)

# Dataset Preparation
class TextDataset(Dataset):
    def __init__(self, tokenizer, data, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        for text in data['text']:
            tokenized_text = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
            self.examples.append(tokenized_text['input_ids'].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_data = TextDataset(tokenizer, dataset['train'], block_size)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# Load Model
model = AutoModelForCausalLM.from_pretrained(model_name)

if peft:
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=['q', 'v'],  # Specific to certain architectures; modify for others
        dtype=torch.float16
    )
    model = get_peft_model(model, config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training Loop
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")

# Save Model
os.makedirs(save_model_path, exist_ok=True)
model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)

print(f"Model saved to {save_model_path}")
