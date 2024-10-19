from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

base_dir = 'human/deberta'
data_dir = 'human/data'
output_dir = 'human/deberta/output'
pretrained_model_name_or_path = 'microsoft/deberta-v3-base'
batch_size = 4

tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_model_name_or_path)
def tokenize(examples):
    return tokenizer(examples['essay'], padding="max_length", truncation=True, max_length=512)

model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=2)

