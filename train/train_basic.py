from transformers import AutoConfig, AutoModel

from datasets import load_dataset
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


from utils import print_params
from prodigyopt import Prodigy
import torch
import argparse
import wandb

from configs import small_config, tiny_config

config = tiny_config

device = "cuda"

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

# Initialize the model with the new configuration
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
# model.to(device)

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
# dataset = load_dataset("GAIR/MathPile", streaming=True)

###
# dataset code
###
print("Loading dataset")


# dataset = load_dataset("cerebras/SlimPajama-627B", streaming=True)
# [Change] Using a map function to format the dataset lazily
def format_dataset(example):
    return {
        "text": f"Abstract:\n{example['abstract']}\n\nArticle:\n{example['article']}"
    }


def tokenize_dataset(item):
    text = item["text"]
    # print(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # print("= inputs =")
    # print(inputs)

    # inputs["labels"] = inputs["input_ids"].clone()
    return {"input_ids": inputs["input_ids"], "labels": inputs["input_ids"].clone()}


dataset = load_dataset("scientific_papers", "arxiv")
dataset = dataset["train"]
dataset = dataset.select(range(10))
# dataset = dataset.map(format_dataset).map(
#     tokenize_dataset,
#     # batched=True,
#     # batch_size=20,
#     num_proc=14,
#     # new_fingerprint="tokenized",
#     # load_from_cache_file=True,
# )

dataset = dataset.with_transform(lambda x: tokenize_dataset(format_dataset(x)))


def collate_data(batch):
    # [Change] Stacking tensors along a new dimension instead of concatenating
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)  # BxS
    labels = torch.stack([item["labels"] for item in batch], dim=0)  # BxS
    print("labels shape, size", labels.shape, labels.size())
    print("input_ids shape, size", input_ids.shape, input_ids.size())
    return {"input_ids": input_ids, "labels": labels}


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_data,
)


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

wandb.init(project="tiny-moe")

step = 0

# [Change] Updated the train loop to use the text field
for epoch in range(args.epochs):
    model.train()

    for batch in train_dataloader:
        # Get text and move to device
        # Get text and convert it to tokens
        print(batch.items())
        # dtype = torch.bfloat16
        in_debug = {name: tensor.dtype for name, tensor in batch.items()}
        in_device_debug = {name: tensor.device for name, tensor in batch.items()}
        print("debug dtype", in_debug)
        print("debug dtype", in_device_debug)

        inputs = {name: tensor.to(device) for name, tensor in batch.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})

        if step % args.save_every_n_steps == 0:
            model.save_pretrained("./models/" + "step" + "_" + str(step))

        if step % args.sample_every_n_steps == 0:
            sample = sample_validation(model, tokenizer, fake_abstract_test)
            wandb.log({"sample": sample})

        step += 1
