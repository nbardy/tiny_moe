from transformers import AutoConfig, AutoModel

from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule
from composer.models import HuggingFaceModel


from utils import print_params
from prodigyopt import Prodigy
import torch
import argparse
import wandb

from configs import small_config, tiny_config

config = tiny_config

device = "cuda"
dtype = torch.bfloat16

# Define the argument parser for command line options
parser = argparse.ArgumentParser(description="Training script for vision models.")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
parser.add_argument(
    "--optimizer",
    type=str,
    default="prodigy",
    choices=["adam", "prodigy"],
    help="Optimizer for training.",
)
# lr
parser.add_argument(
    "--lr", type=float, default=1e-6, help="Learning rate for training."
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps",
)
parser.add_argument(
    "--warmup_steps", type=int, default=0, help="Number of warmup steps"
)
# sample_every_n_steps
parser.add_argument(
    "--sample_every_n_steps", type=int, default=1000, help="Sample test prompt"
)
# save_every_n_steps
parser.add_argument("--save_every_n_steps", type=int, default=5000, help="Save model")

# Parse the arguments
args = parser.parse_args()

# Specify the new configuration

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")

# setattr(small_config, "_attn_implementation", "flash_attention_2")

from composer import ComposerModel


# Initialize the model with the new configuration
class ComposerModelForCausalLM(ComposerModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):  # batch is the output of the dataloader
        # specify how batches are passed through the model
        outputs = self.model(**batch)
        return outputs

    def loss(self, outputs, batch):
        return outputs.loss


# Initialize the Composer model with the new configuration
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

model.to(
    device, dtype=dtype
)  # Moves model to the specified device, which is 'cuda' in this case

model = ComposerModelForCausalLM(model)
# model = HuggingFaceModel(model, tokenizer=tokenizer)

import streaming

from torch.utils.data import DataLoader


# debugger
# import ipdb
# ipdb.set_trace()

# model._attn_implementation
# model.flash

# Choose the optimizer based on the parsed argument
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=0.001
    )  # [Change] Corrected args.learning_rate to args.lr
    lr_scheduler_type = "cosine"
    # Cosine annealing
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs
    )  # [Change] Added num_training_steps to lr_scheduler
elif args.optimizer == "prodigy":
    optimizer = Prodigy(model.parameters(), lr=1.0, weight_decay=0.001)
    lr_scheduler = get_constant_schedule(optimizer)
    print("Learning rate is ignored for Prodigy optimizer")

# calculate to
print_params(model, config)

# Load the dataset

###
# dataset code
##

dataset = load_dataset("GAIR/MathPile", streaming=True)
# dataset = load_dataset("cerebras/SlimPajama-627B", streaming=True)
# dataset = load_dataset("scientific_papers", "arxiv", streaming=True)


# def format_dataset(example):
#     return {
#         "text": f"Abstract:\n{example['abstract']}\n\nArticle:\n{example['article']}"
#     }


def format_dataset(example):
    return example


def tokenize_dataset(item):
    text = item["text"]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]

    return {"input_ids": input_ids, "labels": input_ids.clone()}


dataset = dataset["train"].map(format_dataset).map(tokenize_dataset)
# dataset = dataset["train"]
# dataset = streaming.text.StreamingPile(group_method=


def collate_data(batch):
    # [Change] Concatenating tensors along the sequence length dimension instead of stacking
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)  # [bsz*seq_len]
    labels = torch.cat([item["labels"] for item in batch], dim=0)  # [bsz*seq_len]
    print("labels shape, size", input_ids.shape, input_ids.size())
    print("inputs_ids shape, size", labels.shape, labels.size())
    return {"input_ids": input_ids, "labels": labels}


dataloader = DataLoader(dataset, collate_fn=collate_data)


###
# Validation Code
###

fake_abstract_test = "Abstract:\n This is a paper about novel machine learning techniques. We introduce a sparse MoE gate that is 4x more effeceint than"


def sample_validation(model, tokenizer, prompt, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


###
# Training Code
###

print("Starting training...")

from composer import Trainer
import composer

trainer = Trainer(
    model=model,
    train_dataloader=dataloader,
    max_duration=composer.Time.from_sample(1000),
    optimizers=optimizer,
)
trainer.fit()
