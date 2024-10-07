import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk, Dataset
from .model import TransformerForClassification

CACHE_DIR = "cache"

# Device setup
print("Setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (IMDB movie reviews)
print("Loading dataset...")

# Preprocess the dataset
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Check if cached tokenized data exists
if os.path.exists(os.path.join(CACHE_DIR, "train")) and os.path.exists(os.path.join(CACHE_DIR, "test")):
    print("Loading tokenized datasets from cache...")
    train_dataset = load_from_disk(os.path.join(CACHE_DIR, "train"))
    test_dataset = load_from_disk(os.path.join(CACHE_DIR, "test"))
else:
    print("Tokenized datasets not found in cache. Tokenizing and caching datasets...")

    # Load the dataset from the interwebs or cache
    dataset = load_dataset("imdb")

    # Tokenize the dataset
    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    test_dataset = dataset["test"].map(tokenize_function, batched=True)

    # Save tokenized datasets to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(CACHE_DIR, "train"))
    test_dataset.save_to_disk(os.path.join(CACHE_DIR, "test"))

# Set format for PyTorch and remove unnecessary columns
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader setup
print("Setting up DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the model
model = TransformerForClassification().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop with tqdm progress bar
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        # Set up tqdm progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(tepoch):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].float().unsqueeze(1).to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update total loss
                total_loss += loss.item()

                # Update tqdm progress bar with additional information
                eta = (len(train_loader) - batch_idx) / max(1, tepoch.n / (time.time() - start_time))
                tepoch.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "it/s": f"{tepoch.n / max(1, time.time() - start_time):.2f}",
                    "ETA": f"{eta:.2f}s",
                    "% done": f"{100 * (batch_idx + 1) / len(train_loader):.2f}%"
                })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        checkpoint_path = f"bert_sentiment_checkpoint_epoch_{epoch+1}.pth"
        print("Saving checkpoint to:", checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)

# Train the model
print("Training model...")
train_model(model, train_loader, optimizer, criterion, device)

# Save the model after training
print("Saving model...")
model_path = "bert_sentiment_model.pth"
model_path = os.path.join(os.path.dirname(__file__), model_path)
torch.save(model.state_dict(), model_path)

