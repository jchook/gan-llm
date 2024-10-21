import sys
import torch
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import numpy as np
from .model import data_dir, output_dir, load_deberta_model, get_deberta_lora, load_and_prepare_dataset

# Hyperparameters
batch_size = 4
num_epochs = 1

# Base model and tokenizer
print("Loading DeBERTa model...")
deberta_model, tokenizer = load_deberta_model()

print("Applying LoRA to DeBERTa model...")
model = get_deberta_lora(deberta_model)
model.print_trainable_parameters()

# Dataloader shorthand
print("Loading and preparing dataset...")
train_dataloader = load_and_prepare_dataset(f'{data_dir}/train.csv', tokenizer, batch_size)

# Optimizer and learning rate scheduler
optimizer = AdamW(deberta_model.parameters(), lr=2e-5)
num_training_steps = (num_epochs * len(train_dataloader))
lr_scheduler = get_scheduler(
  name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

# Move model to GPU
print("Moving model to GPU...")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
print("Starting training loop...")
progress_bar = tqdm(range(num_training_steps))
smoothed_accuracy = []

try:
  for epoch in range(num_epochs):
    model.train()

    for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}

      # Forward pass
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
      smoothed_accuracy.append(accuracy)

      if len(smoothed_accuracy) > 100:
        smoothed_accuracy.pop(0)
        smooth_acc = np.mean(smoothed_accuracy)

        progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Smoothed Acc: {smooth_acc:.4f}")
        progress_bar.update(1)

    # Save model at the end of each epoch
    model.save_pretrained(f"{output_dir}/lora_epoch_{epoch+1}")

except Exception as e:
  print(f"Error occurred: {e}")
  torch.cuda.empty_cache()  # Free up VRAM

