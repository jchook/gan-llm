from datasets import load_dataset
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
from peft import TaskType, LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader

base_dir = 'human/deberta_lora'
data_dir = 'human/data'
output_dir = f'{base_dir}/output'
pretrained_model_name_or_path = 'microsoft/deberta-v3-base'
batch_size = 4

# Start here. If the performance is worse than FFT, increase rank to 16, 32, etc
default_lora_config = LoraConfig(
  r=8,
  lora_alpha=16,
  task_type=TaskType.SEQ_CLS,
  inference_mode=False,
  #target_modules=["deberta.encoder.layer.*.attention.self.query_proj", "deberta.encoder.layer.*.attention.self.key_proj", "deberta.encoder.layer.*.attention.self.value_proj"],
  lora_dropout=0.1,
  # bias="none",
)

def load_deberta_model(model_name='microsoft/deberta-v3-base'):
  """
  Load a pre-trained DeBERTa model and tokenizer
  """
  model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return model, tokenizer

def get_deberta_lora(deberta_model, lora_config = default_lora_config):
  """
  Apply LoRA to a pre-trained DeBERTa model
  """
  model = get_peft_model(deberta_model, lora_config)
  return model

def load_and_prepare_dataset(data_file, tokenizer, batch_size):
  """
  Helper function to load and prepare a CSV dataset for training or testing
  """
  def tokenize_fn(examples):
    return tokenizer(examples['essay'], truncation=True, padding='max_length', max_length=512)
  dataset = load_dataset('csv', data_files={'train': data_file})
  # dataset['train'] = dataset['train'].select(range(2000))
  tokenized_datasets = dataset.map(tokenize_fn, batched=True)
  tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
  tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
  return DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)


def load_deberta_lora(lora_dir):
  """
  Load a pre-trained DeBERTa LoRA model
  """
  return AutoPeftModelForSequenceClassification.from_pretrained(lora_dir)