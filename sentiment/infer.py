import os
import torch
from transformers import BertTokenizer
from .model import TransformerForClassification

# Load the trained model
model_path = "bert_sentiment_model.pth"
model_path = os.path.join(os.path.dirname(__file__), model_path)
model = TransformerForClassification()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to perform inference
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to the device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # Output is a probability (sigmoid), so we classify based on a threshold of 0.5
    probability = output.item()
    sentiment = "positive" if probability >= 0.5 else "negative"

    return sentiment, probability

# Example usage
if __name__ == "__main__":
    text = ""
    while text != "exit":
      # Get user input
      text = input("Enter a movie review: ")

      # Predict sentiment
      sentiment, probability = predict_sentiment(text)

      # Output result
      print(f"Predicted sentiment: {sentiment} (Probability: {probability:.4f})")

