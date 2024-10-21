import torch
from .model import load_deberta_model, output_dir
from torch.nn.functional import softmax
from peft import PeftModel, PeftConfig

lora_model_dir = f"{output_dir}/lora_epoch_1"

# Load the pre-trained model
deberta_model, tokenizer = load_deberta_model()

# Load the LoRA model
#model = AutoPeftModel.from_pretrained(f"{output_dir}/lora_epoch_1")
#load_peft_model(deberta_model, f"{output_dir}/lora_epoch_1")
model = PeftModel.from_pretrained(deberta_model, lora_model_dir)
model.eval()

# Use GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

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
  input_lines = []
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

