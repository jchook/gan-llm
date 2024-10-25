import torch
from torch.optim import AdamW
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from .model import data_dir, output_dir, load_metric, load_deberta_model, load_and_prepare_dataset

# Hyperparameters
batch_size = 4
num_epochs = 3
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PEFT config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    lora_dropout=0.1,

    # Essential for deBERTa since it does not have a pretrained pooler layer.
    # https://github.com/huggingface/peft/issues/2171#issuecomment-2431727061
    modules_to_save=["classifier", "pooler"],
)

# Base model and tokenizer
print("Loading DeBERTa model...")
model, tokenizer = load_deberta_model()

print("Applying LoRA to DeBERTa model...")
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

        # Compute accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == batch["labels"]).float().mean().item()
        smoothed_accuracy.append(accuracy)
        if len(smoothed_accuracy) > 100:
            smoothed_accuracy.pop(0)
        if step % 100 == 0:
            print(f"epoch {epoch}, step {step}: accuracy={sum(smoothed_accuracy) / len(smoothed_accuracy)}")

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
    print(f"epoch {epoch}:", eval_metric)

    # Reset the metric after each epoch
    #metric.reset()

    # Save the model after each epoch
    model.save_pretrained(f"{output_dir}/lora_epoch_{epoch+1}")

# Save the final model
model.save_pretrained(f"{output_dir}/lora_final")
