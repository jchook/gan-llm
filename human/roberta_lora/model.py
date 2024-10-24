from datasets import load_dataset
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
from peft import TaskType, LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
import evaluate

base_dir = 'human/roberta_lora'
data_dir = 'human/data'
output_dir = f'{base_dir}/output'
pretrained_model_name_or_path = 'roberta-base'
batch_size = 4

def load_base_model(model_name=pretrained_model_name_or_path):
  """
  Load a pre-trained DeBERTa model and tokenizer
  """
  model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return model, tokenizer


def clean_text_data(input: str):
  """
  Clean text data by removing special characters and extra spaces
  """
  return ' '.join(input.split())


def load_and_prepare_dataset(data_file, tokenizer, batch_size):
  """
  Helper function to load and prepare a CSV dataset for training or testing
  """
  def tokenize_fn(examples):
    examples['essay'] = [clean_text_data(essay) for essay in examples['essay']]
    return tokenizer(examples['essay'], truncation=True, padding='max_length', max_length=512)
  dataset = load_dataset('csv', data_files={'train': data_file})
  # dataset['train'] = dataset['train'].select(range(2000))
  tokenized_datasets = dataset.map(tokenize_fn, batched=True)
  tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
  tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
  return DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)


def load_metric():
  return CompositeMetric(["accuracy", "f1"])

class CompositeMetric():
  def __init__(self, metric_names):
    # Initialize the parent class with some required information
    super().__init__()
    # Load metrics from evaluate library
    self.metrics = {name: evaluate.load(name) for name in metric_names}

  def _info(self):
    # This is a placeholder MetricInfo object.
    # You can customize the description or other attributes as needed.
    return evaluate.MetricInfo(
        description="A composite metric for aggregating multiple metrics (e.g., accuracy, F1)",
        citation="",
        inputs_description="Predictions and references for multiple metrics.",
        features=[],
        homepage="",
        license="",
        codebase_urls=["https://github.com/huggingface/evaluate"],
    )

  def add_batch(self, predictions, references, **kwargs):
    # Add batch to each metric
    for metric in self.metrics.values():
      metric.add_batch(predictions=predictions, references=references, **kwargs)

  def compute(self):
    # Compute all metrics and return them as a dictionary
    results = {}
    for name, metric in self.metrics.items():
      if name == "f1":
        results[name] = metric.compute(average='weighted')
      else:
        results[name] = metric.compute()
    return results

  def reset(self):
    # Reset all metrics
    for metric in self.metrics.values():
      metric.reset()

