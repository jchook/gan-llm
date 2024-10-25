from transformers import DebertaV2ForSequenceClassification
from .temperature_scaling import ModelWithTemperature
from .model import load_deberta_model, data_dir, output_dir, load_and_prepare_dataset
from peft import PeftModel, PeftConfig

lora_model_dir = f"{output_dir}/lora_epoch_3"

# Load the pre-trained model
model, tokenizer = load_deberta_model()

# Load the LoRA model
model = PeftModel.from_pretrained(model, lora_model_dir)
model = ModelWithTemperature(model)

dataloader = load_and_prepare_dataset(f"sources/hn/data/01/validation.csv", tokenizer, 4)

model.set_temperature(dataloader)
print(model.temperature.item())
