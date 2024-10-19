import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2ForSequenceClassification
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .model import batch_size, output_dir, data_dir, tokenize

# Load the pre-trained model
model = DebertaV2ForSequenceClassification.from_pretrained(f"{output_dir}/model_epoch_1")

# Load the dataset and tokenize
dataset = load_dataset('csv', data_files={'test': f'{data_dir}/test.csv'})
tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize DataLoader
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=1)

# Move model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Initialize metrics storage
all_predictions = []
all_labels = []

# Evaluation loop
model.eval()
progress_bar = tqdm(eval_dataloader, desc="Evaluating")

try:
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.pop("label")  # Extract the labels
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)

            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits

            # Predictions and labels
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

except Exception as e:
    print(f"Error occurred during evaluation: {e}")
    torch.cuda.empty_cache()  # Free up VRAM if an error occurs

# Compute evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

