from transformers import AutoConfig, AutoModel

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule

from utils import print_params

from prodigyopt import Prodigy

import torch

import argparse

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
parser.add_argument("--batch_size", type=float, default=16, help="Batch size")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=float,
    default=1,
    help="Number of gradient accumulation steps",
)

# Parse the arguments
args = parser.parse_args()


# Specify the new configuration
new_config = AutoConfig.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base", trust_remote_code=True
)
# [Change] Converted configuration settings into a dictionary
# Apply the updates to the new_config
for key, value in config_tiny.items():
    setattr(new_config, key, value)


# Initialize the model with the new configuration
model = AutoModelForCausalLM.from_config(new_config, trust_remote_code=True)
model.to("cuda")

# Choose the optimizer based on the parsed argument
warmup_steps = None
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.001
    )
    lr_scheduler_type = "cosine"
    # Cosine annealing
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps)
elif args.optimizer == "prodigy":
    optimizer = Prodigy(model.parameters(), lr=1.0, weight_decay=0.001)
    lr_scheduler = get_constant_schedule(optimizer)
    print("Learning rate is ignored for Prodigy optimizer")

# calculate to
print_params(model, new_config)

# Load the dataset
# dataset = load_dataset("GAIR/MathPile", streaming=True)

print("Loading dataset")

# dataset = load_dataset("cerebras/SlimPajama-627B", streaming=True)
dataset = load_dataset("scientific_papers", "arxiv")


def format_dataset(example):
    return {
        "text": f"Abstract:\n{example['abstract']}\n\nArticle:\n{example['article']}"
    }


dataset = load_dataset("scientific_papers", "arxiv")
dataset = dataset.map(format_dataset)  # This is a lazy operation
train_dataset = dataset["train"]


# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=args.epochs,  # total number of training epochs
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
    logging_dir="./logs",  # directory for storing logs
    report_to="wandb",
    max_steps=1000,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
)

print("Starting training...")


# Initialize the Trainer
trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=dataset,  # training dataset
    optimizers=(optimizer, lr_scheduler),
    max_steps=
)

# Train the model
trainer.train()
