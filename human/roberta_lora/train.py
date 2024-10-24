import torch
import numpy as np
from torch.optim import AdamW
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from .model import data_dir, output_dir, load_metric, load_base_model, load_and_prepare_dataset

# Hyperparameters
batch_size = 32
num_epochs = 3
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PEFT config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    inference_mode=False,
)

# Base model and tokenizer
print("Loading base model...")
model, tokenizer = load_base_model()

print("Applying LoRA to base model...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.compile()

# Dataloader shorthand
print("Loading and preparing dataset...")
train_dataloader = load_and_prepare_dataset(f'{data_dir}/train.csv', tokenizer, batch_size)
eval_dataloader = load_and_prepare_dataset(f'{data_dir}/validation.csv', tokenizer, batch_size)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = (num_epochs * len(train_dataloader))

# Learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# Metrics
metric = load_metric()

# Move model to GPU
print(f"Moving model to {device}...")
model.to(device)

# Training loop
print("Starting training loop...")
smoothed_accuracy = []
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Compute accuracy during training
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = (predictions == batch["labels"]).float().mean().item()
        smoothed_accuracy.append(accuracy)
        if len(smoothed_accuracy) > 100:
            smoothed_accuracy.pop(0)
        if (step + 1) % 100 == 0:
            smooth_acc = np.mean(smoothed_accuracy)
            print(f" acc {smooth_acc}")

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(predictions=predictions, references=references)

    # Compute and print metrics
    eval_metric = metric.compute()
    print(f"epoch {epoch+1}:", eval_metric)

    # Save the model after each epoch
    model.save_pretrained(f"{output_dir}/lora_epoch_{epoch+1}")

