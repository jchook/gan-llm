import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .model import output_dir

model_dir = f"{output_dir}/model_epoch_3"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the LoRA model
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

model.eval()
model.to(device)

class_to_label = {0: "Human", 1: "Machine"}

# Function to perform inference
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
    inputs.to(device)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    confidence_score = outputs.logits.softmax(dim=-1)[0, predicted_class].item()
    return f"Predicted Class: {class_to_label[predicted_class]}, Confidence: {confidence_score:.4f}"

# Define Gradio interface
demo = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs="text",
    title="LoRA Model Inference",
    description="Enter text to classify using the fine-tuned LoRA adapter.",
)

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=True)


