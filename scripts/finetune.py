import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ======= Set Random Seeds ======= #
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

# ======= Hyperparameters ======= #
MODEL_NAME = 'gpt2'
DATA_PATH = "data/user_twit_bios.csv"
SAVE_MODEL_PATH = "./finetuned_gpt2"
MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======= Load Dataset ======= #
class TextDataset(Dataset):
    """
    Custom Dataset class for fine-tuning GPT-2 on text data.
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

# ======= Load and Preprocess Data ======= #
def load_and_preprocess_data(file_path, test_size=0.2):
    """
    Load the CSV file, split into train/validation sets, and return dataframes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path)
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data

# ======= Initialize Model ======= #
def initialize_model(model_name, tokenizer_name):
    """
    Load the GPT-2 model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# ======= Training Loop ======= #
def train_model(
    model, train_loader, val_loader, optimizer, scheduler, num_epochs, device
):
    """
    Training loop for fine-tuning GPT-2.
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
    # Load and preprocess data
    train_data, val_data = load_and_preprocess_data(DATA_PATH)

    # Initialize tokenizer and model
    tokenizer, model = initialize_model(MODEL_NAME, MODEL_NAME)

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
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    train_model(model, train_loader, val_loader, optimizer, scheduler, NUM_EPOCHS, DEVICE)
