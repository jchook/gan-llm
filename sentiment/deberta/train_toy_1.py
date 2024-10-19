import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
from datasets import load_dataset
import torch.nn.functional as F
from tqdm import tqdm

# Initialize the tokenizer and the model
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
model = DebertaV2ForSequenceClassification.from_pretrained(
    'microsoft/deberta-v2-xlarge',
    num_labels=2,
    problem_type="single_label_classification"
)

# Hyperparameters
batch_size = 2
learning_rate = 2e-5
epochs = 3

# Load dataset and tokenize
def tokenize_function(examples):
    return tokenizer(examples['essay'], truncation=True, padding='max_length', max_length=512)

# Load dataset from csv and tokenize
dataset = load_dataset("csv", data_files={'train': 'data/train.csv'})
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=4, shuffle=True)

# Convert dataset to torch tensors and DataLoader
# input_ids = torch.tensor(tokenized_datasets['train']['input_ids'])
# attention_mask = torch.tensor(tokenized_datasets['train']['attention_mask'])
# labels = torch.tensor(tokenized_datasets['train']['label'])

# Create TensorDataset and DataLoader
# train_dataset = TensorDataset(input_ids, attention_mask, labels)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    correct_predictions = 0
    num_samples = 0

    for batch in tqdm(train_dataloader):
        # Move batch to device
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        total_loss += loss.item()
        predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct_predictions += (predictions == labels).sum().item()
        num_samples += labels.size(0)

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / num_samples

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

print("Training complete!")

