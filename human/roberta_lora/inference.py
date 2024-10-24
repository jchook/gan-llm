import torch
from .model import load_base_model, output_dir
from torch.nn.functional import softmax
from peft import PeftModel, PeftConfig

lora_model_dir = f"{output_dir}/lora_epoch_1"

# Load the pre-trained model
base_model, tokenizer = load_base_model()

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, lora_model_dir)
model.eval()

# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a function to process and evaluate text
def evaluate_text(text):
  inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

    probabilities = softmax(logits, dim=-1)
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
    predictions = evaluate_text(text)
    print(predictions)

    # Print the prediction result
    #print(f"\nPredicted class: {predicted_class}, Confidence score: {confidence_score:.4f}")

if __name__ == "__main__":
  repl()


