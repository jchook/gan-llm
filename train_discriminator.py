import os
import time
import torch
from torch._prims_common import check
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from datasets import load_dataset, load_from_disk, Dataset
from .model import RobertaDiscriminator

CACHE_DIR = "cache"
CACHE_PREFIX = "essays_"

checkpoint_path = "roberta_discriminator_model.pth"

# Device setup
print("Setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess the dataset
print("Loading tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")

def tokenize(examples):
    return tokenizer(examples["text"])

# Check if cached tokenized data exists
if os.path.exists(os.path.join(CACHE_DIR, "train")) and os.path.exists(os.path.join(CACHE_DIR, "test")):
    print("Loading tokenized datasets from cache...")
    train_dataset = load_from_disk(os.path.join(CACHE_DIR, f"{CACHE_PREFIX}train"))
    test_dataset = load_from_disk(os.path.join(CACHE_DIR, f"{CACHE_PREFIX}test"))
else:
    print("Tokenized datasets not found in cache. Tokenizing and caching datasets...")

    # Load dataset (IMDB movie reviews)
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files={'train': 'data/train.csv', 'test': 'data/validation.csv'})

    # Tokenize the dataset
    train_dataset = dataset["train"].map(tokenize, batched=True)
    test_dataset = dataset["test"].map(tokenize, batched=True)

    # Save tokenized datasets to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(CACHE_DIR, f"{CACHE_PREFIX}train"))
    test_dataset.save_to_disk(os.path.join(CACHE_DIR, f"{CACHE_PREFIX}test"))

# Set format for PyTorch and remove unnecessary columns
train_dataset.set_format(type="torch", columns=["essay_id", "essay", "origin"])
test_dataset.set_format(type="torch", columns=["essay_id", "essay", "origin"])

# DataLoader setup
print("Setting up DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the model
model = RobertaDiscriminator().to(device)

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
                labels = batch["origin"].float().unsqueeze(1).to(device)

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
        checkpoint_epoch_path = f"{checkpoint_path}_{epoch+1}.pth"
        print("Saving checkpoint to:", checkpoint_epoch_path)
        torch.save(model.state_dict(), checkpoint_epoch_path)

# Train the model
print("Training model...")
train_model(model, train_loader, optimizer, criterion, device)

# Save the model after training
print("Saving model...")
torch.save(model.state_dict(), checkpoint_path)

