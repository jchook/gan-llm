import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from torch.nn.functional import softmax
from .model import tokenizer, output_dir  # Assuming you have model's output directory configured here

# Load the pre-trained model and tokenizer
model = DebertaV2ForSequenceClassification.from_pretrained(f"{output_dir}/model_epoch_1")

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Define a function to process and evaluate text
def evaluate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = softmax(logits, dim=-1)

        # Get the predicted class and confidence score
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence_score = probabilities[0, predicted_class].item()

    return predicted_class, confidence_score


# A simple REPL for text input
print("Enter multiline text (Ctrl+D to evaluate or 'exit' to quit):")

def repl():
    while True:
        try:
            print("\nInput text:")
            # Read multiline input
            input_lines = []
            while True:
                line = input()
                if line.strip().lower() == 'exit':
                    print("Exiting.")
                    return
                input_lines.append(line)

        except EOFError:
            pass  # Ctrl+D is handled here to terminate multiline input

        # Join the input lines and evaluate the text
        text = "\n".join(input_lines)
        predicted_class, confidence_score = evaluate_text(text)

        # Print the prediction result
        print(f"\nPredicted class: {predicted_class}, Confidence score: {confidence_score:.4f}")

if __name__ == "__main__":
    repl()

