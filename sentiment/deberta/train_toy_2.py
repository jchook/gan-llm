import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AdamW, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

# Load dataset and tokenizer
dataset = load_dataset('imdb')
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')

# Tokenize dataset with max_length=512
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=2, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=2)

# Load pre-trained DebertaV2ForSequenceClassification
model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge', num_labels=2)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

# Move model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
progress_bar = tqdm(range(num_training_steps))
smoothed_accuracy = []

try:
    for epoch in range(num_epochs):
        model.train()
        running_accuracy = []

        for batch in train_dataloader:
            batch["labels"] = batch.pop("label")  # Rename "label" to "labels"
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Compute accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch["labels"]).float().mean().item()
            running_accuracy.append(accuracy)
            smoothed_accuracy.append(accuracy)

            if len(smoothed_accuracy) > 100:
                smoothed_accuracy.pop(0)
            smooth_acc = np.mean(smoothed_accuracy)

            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Smoothed Acc: {smooth_acc:.4f}")
            progress_bar.update(1)

        # Save model at the end of each epoch
        model.save_pretrained(f"model_epoch_{epoch+1}")

except Exception as e:
    print(f"Error occurred: {e}")
    torch.cuda.empty_cache()  # Free up VRAM

