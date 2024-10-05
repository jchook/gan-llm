import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TransformerForClassification(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super(TransformerForClassification, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # Classification head: maps pooled BERT output to a binary classification (real/fake)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token's hidden state as the pooled output
        pooled_output = outputs.pooler_output

        # Pass through the classification head
        return self.classifier(pooled_output)

# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TransformerForClassification().to(device)

# Example input
text = "This is an example sentence for classification."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
output = model(**inputs.to(device))
print(output)

