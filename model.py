import torch.nn as nn
from transformers import RobertaForSequenceClassification, LlamaModel

class RobertaDiscriminator(nn.Module):
    def __init__(self, pretrained_model_name=""):
        super(RobertaDiscriminator, self).__init__()

        # Load the pre-trained model from Hugging Face
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name)

        # Classification head: transforming RoBERTa's pooled output to a real/fake classification
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        # Get the output from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Pool the output (mean pooling)
        pooled_output = outputs.pooler_output

        # Apply classification head to determine if real or fake
        return self.classifier(pooled_output)


class LlamaGenerator(nn.Module):
    def __init__(self, pretrained_model_name="huggingface/llama-3.2-3b"):
        super(LlamaGenerator, self).__init__()
        self.model = LlamaModel.from_pretrained(pretrained_model_name)

