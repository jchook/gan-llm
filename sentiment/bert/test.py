import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk
from .model import TransformerForClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Constants
CACHE_DIR = "cached_test_data"
BATCH_SIZE = 16

# Load the trained model
model_path = "bert_sentiment_model.pth"
model_path = os.path.join(os.path.dirname(__file__), model_path)
model = TransformerForClassification()
model.load_state_dict(torch.load(model_path))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Check if cached test dataset exists
if os.path.exists(CACHE_DIR):
    print("Loading cached vectorized test dataset...")
    test_dataset = load_from_disk(CACHE_DIR)
else:
    print("Cached dataset not found. Loading and processing raw dataset...")

    # Load raw test dataset
    dataset = load_dataset("imdb")
    test_dataset = dataset["test"]

    # Preprocess the test dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # Tokenize the test dataset and cache it
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Cache the processed dataset
    test_dataset.save_to_disk(CACHE_DIR)
    print(f"Processed dataset cached at {CACHE_DIR}")

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to evaluate the model with progress bar
def evaluate(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []
    criterion = nn.BCELoss()  # Loss function for binary classification
    total_loss = 0

    # tqdm progress bar
    with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().unsqueeze(1).to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Convert outputs to binary predictions (0 or 1)
            preds = (outputs >= 0.5).float()

            # Store predictions and true labels
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Update the progress bar with the current average loss
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix(loss=avg_loss)

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return total_loss / len(test_loader), accuracy, f1

# Run evaluation
avg_loss, accuracy, f1 = evaluate(model, test_loader, device)

# Print final results
print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

