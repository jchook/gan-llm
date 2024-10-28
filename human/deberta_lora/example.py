import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, save_peft_model, load_peft_model
from torch.optim import AdamW

def load_base_model(model_name, lora_config):
  # Load pre-trained model and tokenizer
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # Apply LoRA
  model = get_peft_model(model, lora_config)
  return model, tokenizer

# Configuration for LoRA
lora_config = LoraConfig(
  r=8,
  lora_alpha=32,
  target_modules=["query", "value"],  # Attention layers to target for LoRA
  lora_dropout=0.1,
  bias="none",
)

# Model loading
base_model, tokenizer = load_base_model("microsoft/deberta-v3-base", lora_config)

# Training loop example
def train_loop(model, tokenizer, train_dataloader, optimizer, device):
  model.train()
  for epoch in range(num_epochs):
    for batch in train_dataloader:
      inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True, max_length=512)
      inputs = {key: value.to(device) for key, value in inputs.items()}
      labels = batch["labels"].to(device)

      # Forward pass
      outputs = model(**inputs, labels=labels)
      loss = outputs.loss

      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"Epoch {epoch}, Loss: {loss.item()}")


# Example optimizer setup
optimizer = AdamW(base_model.parameters(), lr=5e-5)

# Assuming train_dataloader is defined and num_epochs is set
num_epochs = 1
train_dataloader = []  # Placeholder: Replace with actual DataLoader

# Moving model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)

# Training loop
train_loop(base_model, tokenizer, train_dataloader, optimizer, device)

# Saving LoRA weights
save_peft_model(base_model, "./lora_deberta_weights")

# Loading LoRA weights for inference
base_model = load_peft_model(base_model, "./lora_deberta_weights")
base_model.to(device)

# Testing tokenizer output
text = "This is a sample input to test LoRA fine-tuning in DeBERTa model."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Forward pass for inference
inputs = {key: value.to(device) for key, value in inputs.items()}
out = base_model(**inputs)
print(out)

# NOTE:
# This setup configures the model for LoRA-based fine-tuning, targeting only specific attention layers for efficiency.
# The training loop modifies only the low-rank adaptation layers.
# The LoRA weights are saved separately, allowing for efficient storage and reusability.

