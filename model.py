import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer

class LlamaDiscriminator(nn.Module):
    def __init__(self, pretrained_model_name="huggingface/llama-3.1-7b"):
        super(LlamaDiscriminator, self).__init__()

        # Load the pre-trained Llama model from Hugging Face
        self.llama = LlamaModel.from_pretrained(pretrained_model_name)

        # Classification head: transforming Llama's pooled output (mean of embeddings) to a real/fake classification
        self.classifier = nn.Sequential(
            nn.Linear(self.llama.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        # Get the output from the Llama model
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)

        # Pool the output (mean pooling)
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        # Apply classification head to determine if real or fake
        return self.classifier(pooled_output)

# Example usage
pretrained_model_name = "huggingface/llama-3.1-7b"
discriminator = LlamaDiscriminator(pretrained_model_name).to(device)

# Tokenizer for input
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name)
inputs = tokenizer("This is a test sentence.", return_tensors="pt")

# Run through discriminator
outputs = discriminator(**inputs.to(device))
print(outputs)
