import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from model import TransformerForClassification

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (IMDB movie reviews)
dataset = load_dataset("imdb")

# Preprocess the dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize datasets
train_dataset = dataset["train"].map(preprocess_function, batched=True)
test_dataset = dataset["test"].map(preprocess_function, batched=True)

# Remove unnecessary columns and set input format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the model
model = TransformerForClassification().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
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

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Train the model
train_model(model, train_loader, optimizer, criterion, device)

# Save the model after training
torch.save(model.state_dict(), "bert_sentiment_model.pth")

