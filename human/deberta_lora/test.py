import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .model import load_and_prepare_dataset, data_dir, load_deberta_model, output_dir
from peft import PeftModel, PeftConfig

lora_model_dir = f"{output_dir}/lora_epoch_1"

# Load the pre-trained model
deberta_model, tokenizer = load_deberta_model()

# Load the LoRA model
#model = AutoPeftModel.from_pretrained(f"{output_dir}/lora_epoch_1")
#load_peft_model(deberta_model, f"{output_dir}/lora_epoch_1")
model = PeftModel.from_pretrained(deberta_model, lora_model_dir)
model.eval()

# Compile the model
model.compile()

# Load the dataset and tokenize
eval_dataloader = load_and_prepare_dataset(f'{data_dir}/test.csv', tokenizer, batch_size=4)

# Move model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Initialize metrics storage
all_predictions = []
all_labels = []

# Evaluation loop
model.eval()
progress_bar = tqdm(eval_dataloader, desc="Eval")
smoothed_accuracy = []

try:
  with torch.no_grad():
    for batch in progress_bar:
      labels = batch['labels']
      batch = {k: v.to(device) for k, v in batch.items()}
      labels = labels.to(device)

      # Forward pass
      outputs = model(**batch)
      logits = outputs.logits

      # Predictions and labels
      predictions = torch.argmax(logits, dim=-1)
      all_predictions.extend(predictions.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

      accuracy = (predictions == batch["labels"]).float().mean().item()
      smoothed_accuracy.append(accuracy)

      if len(smoothed_accuracy) > 100:
        smoothed_accuracy.pop(0)
        smooth_acc = np.mean(smoothed_accuracy)
        progress_bar.set_description(f"Loss: {outputs.loss.item():.4f}, Smoothed Acc: {smooth_acc:.4f}")


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

