import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# ======= Set Random Seeds ======= #
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

# ======= Hyperparameters ======= #
MODEL_NAME = 'gpt2'
DATA_PATH = "data/user_twit_bios.csv"
SAVE_MODEL_PATH = "./finetuned_gpt2_lora"
MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA Config
USE_PEFT = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ======= Load Dataset ======= #
class TextDataset(Dataset):
    """
    Custom Dataset class for fine-tuning GPT-2 on text data with PEFT.
    """
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }


# ======= Data Preprocessing ======= #
def load_and_preprocess_data(file_path, test_size=0.2):
    """
    Load and split data into training and validation sets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path)
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data


# ======= Model Initialization ======= #
def initialize_model_with_peft(model_name, use_peft, lora_config):
    """
    Initialize the GPT-2 model with or without PEFT (LoRA).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_peft:
        print("Preparing model for LoRA fine-tuning...")
        model = prepare_model_for_int8_training(model)
        peft_config = LoraConfig(
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        print("LoRA configuration applied successfully!")

    return tokenizer, model


# ======= Training Loop ======= #
def train_model(
    model, train_loader, val_loader, optimizer, scheduler, num_epochs, device
):
    """
    Training loop for fine-tuning GPT-2 with LoRA.
    """
    model.to(device)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_model.pth"))
            print("Saved best model!")

    # Plot Losses
    plot_losses(train_losses, val_losses)


def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def plot_losses(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


# ======= Main Script ======= #
if __name__ == "__main__":
    # Prepare directories
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # Load and preprocess data
    train_data, val_data = load_and_preprocess_data(DATA_PATH)

    # Initialize tokenizer and model with PEFT
    lora_config = {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT}
    tokenizer, model = initialize_model_with_peft(MODEL_NAME, USE_PEFT, lora_config)

    # Prepare datasets and dataloaders
    train_dataset = TextDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = TextDataset(val_data, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, NUM_EPOCHS, DEVICE)
