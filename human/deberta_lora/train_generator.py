import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
from .model import device, output_dir
import torch.nn.functional as F

# Hyperparameters
batch_size = 4
num_epochs = 3
learning_rate = 3e-4

# PEFT configuration for Generator
generator_peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_dropout=0.1,
)

_PROMPT_PREFIX = (
    "Assume the role of a Hacker News discussion user. Reply intelligently to "
    "the provided comment in the style of a Hacker News comment section. Reply "
    "using approximately as many words as the provided comment with an original "
    "thought that directly addresses the provided comment. Only output the reply "
    "to the comment. Here is the comment to reply to: \n\n"
)

class Generator:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", peft_config=generator_peft_config):
        # Load base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Apply PEFT LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Move model to GPU
        self.model.to(device)

        # Optimizer and learning rate scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

    def generate_text_samples(self, batch_size):
        # Generate text based on random input prompts or fixed prompt
        prompts = ["Sample prompt"] * batch_size  # Replace with actual prompts if available
        input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Generate text
        generated_ids = self.model.generate(input_ids=input_ids, max_length=500)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts

    def fine_tune(self, train_dataloader, discriminator, num_epochs=num_epochs):
        # Prepare for training
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0.06 * num_training_steps,
            num_training_steps=num_training_steps,
        )

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # Generate text samples from the generator
                generated_texts = self.generate_text_samples(batch_size)

                # Tokenize generated texts with discriminator’s tokenizer
                discriminator_inputs = discriminator.tokenizer(
                    generated_texts, return_tensors="pt", padding=True, truncation=True
                ).to(device)

                # Evaluate with discriminator
                with torch.no_grad():
                    discriminator_outputs = discriminator.model(**discriminator_inputs)
                discriminator_predictions = torch.sigmoid(discriminator_outputs.logits)

                # Inverted ground truth labels (all zeros to represent "real" for generator training)
                inverted_labels = torch.zeros_like(discriminator_predictions).to(device)

                # Calculate generator's loss to make the discriminator believe the generated samples are "real"
                gen_loss = F.binary_cross_entropy(discriminator_predictions, inverted_labels)

                # Backpropagation
                gen_loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()

                # Display training progress
                if step % 100 == 0:
                    print(f"Epoch {epoch}, Step {step}: Generator Loss = {gen_loss.item():.4f}")

            # Save the model after each epoch
            self.model.save_pretrained(f"{output_dir}/generator_epoch_{epoch+1}")

