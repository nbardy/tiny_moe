from datasets import interleave_datasets

import chunk
import random


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

CONTEXT_SIZE = 512

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
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for training.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps",
)
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
# sample_every_n_steps
parser.add_argument("--sample_every_n_steps", type=int, default=1000, help="Sample test prompt")
# save_every_n_steps
parser.add_argument("--save_every_n_steps", type=int, default=5000, help="Save model")

# Parse the arguments
args = parser.parse_args()

# Specify the new configuration

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")

# setattr(small_config, "_attn_implementation", "flash_attention_2")

dtype = torch.bfloat16

# Initialize the model with the new configuration
model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
model.to(device, dtype=dtype)

# debugger
# import ipdb
# ipdb.set_trace()

# model._attn_implementation
# model.flash

# Choose the optimizer based on the parsed argument
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001, fused=True)  # [Change] Corrected args.learning_rate to args.lr
    lr_scheduler_type = "cosine"
    # Cosine annealing
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs
    )  # [Change] Added num_training_steps to lr_scheduler
elif args.optimizer == "prodigy":
    optimizer = Prodigy(model.parameters(), lr=1.0, weight_decay=0.0001)
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


def tokenize_dataset(item):
    text = item["text"]
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Directly return the tokenized text without chunking
    return {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
        "labels": tokenized_text["input_ids"].clone(),
    }


# [Change] Interleaving multiple datasets with specified probabilities using Hugging Face's Datasets library

import random


def format_item_fn(header_columns, prefix="", chunk_size=100, base_key="markdown"):
    def formatter(item):
        base_text = item[base_key]
        markdown_chunks = [" ".join(base_text.split()[i : i + chunk_size]) for i in range(0, len(base_text.split()), chunk_size)]

        indices = list(range(len(markdown_chunks)))
        all_chunks = []

        for chunk in markdown_chunks:
            header_text = " ".join(f"{key}: {item[key]}" for key in header_columns if key in item)
            formatted_text = f"{prefix} {header_text} Excerpt: {chunk}"
            all_chunks.append({"text": formatted_text})

        # Do one chunk with just prefix
        if prefix != "":
            random_chunk = random.choice(markdown_chunks)
            formatted_text = f"{prefix} {random_chunk}"
            all_chunks.append({"text": formatted_text})

        return all_chunks

    return formatter


format_science_papers = format_item_fn(["absract", "article"], prefix="Science paper", base_key="article")
format_wiki = format_item_fn(["title", "categories"], prefix="Wikipedia", base_key="markdown")
format_mathpile_chunks = format_item_fn([], prefix="Math Pile", base_key="text")
format_pile_chunks = format_item_fn([], prefix="Web Text", base_key="text")
format_grounded_textbooks = format_item_fn(["topic", "model", "concepts", "outline"], prefix="Grounded Textbook")
format_synthetic_textbooks = format_item_fn(["model", "topic", "concepts", "outline"], prefix="Synthetic Textbooks")
format_fake_textbooks = format_item_fn(["model", "topic", "outline", "field", "subfield"], prefix="Fake Textbooks(Some Grounded)")
format_orca_textbooks = format_item_fn([], base_key="textbook", prefix="Tiny Orca Textbooks")
format_strange_textbooks = format_item_fn([], prefix="Tiny Strange Textbooks", base_key="text")
format_math_textbooks = format_item_fn([], prefix="Tiny Math Textbooks", base_key="text")
format_code_textbooks = format_item_fn([], prefix="Tiny Code Textbooks", base_key="text")
format_tiny_textbooks = format_item_fn([], prefix="Tiny Textbooks", base_key="text")
format_muse_textbooks = format_item_fn([], prefix="Muse Textbooks", base_key="text")
format_all_you_need_books = format_item_fn([], prefix="Synthetic Code Textbooks", base_key="text")


all_datasets = [
    # 30% web text
    (0.15, load_dataset("cerebras/SlimPajama-627B", streaming=True)["train"].with_transform(format_pile_chunks).flatten()),
    (0.15, load_dataset("JeanKaddour/minipile", streaming=True)["train"].with_transform(format_pile_chunks).flatten()),
    # 20% papers on xaxriv
    (0.18, load_dataset("GAIR/MathPile_Commercial", streaming=True)["train"].with_transform(format_mathpile_chunks).flatten()),
    (0.02, load_dataset("scientific_papers", "arxiv", streaming=True)["train"].with_transform(format_science_papers).flatten()),
    # Rest is sythetic textbooks
    (0.05, load_dataset("open-phi/wile-e", streaming=True)["train"].with_transform(format_synthetic_textbooks).flatten()),
    (0.05, load_dataset("open-phi/textbooks", streaming=True)["train"].with_transform(format_fake_textbooks).flatten()),
    (0.05, load_dataset("open-phi/textbooks_grounded", streaming=True)["train"].with_transform(format_grounded_textbooks).flatten()),
    (0.05, load_dataset("nampdn-ai/tiny-orca-textbooks", streaming=True)["train"].with_transform(format_orca_textbooks).flatten()),
    (0.05, load_dataset("nampdn-ai/tiny-strange-textbooks", streaming=True)["train"].with_transform(format_strange_textbooks).flatten()),
    (0.05, load_dataset("euirim/goodwiki", streaming=True)["train"].with_transform(format_wiki).flatten()),
    (0.06, load_dataset("nampdn-ai/tiny-math-textbooks", streaming=True)["train"].with_transform(format_math_textbooks).flatten()),
    (0.04, load_dataset("nampdn-ai/tiny-code-textbooks", streaming=True)["train"].with_transform(format_code_textbooks).flatten()),
    (0.02, load_dataset("nampdn-ai/tiny-textbooks", streaming=True)["train"].with_transform(format_tiny_textbooks).flatten()),
    (0.07, load_dataset("TanvirOnHF/muse_textbooks", streaming=True)["train"].with_transform(format_muse_textbooks).flatten()),
    # ndurkee/muse_textbooks
    (0.01, load_dataset("ndurkee/muse_textbooks", streaming=True)["train"].with_transform(format_muse_textbooks).flatten()),
    (0.05, load_dataset("SkySyrup/muse_textbooks", streaming=True)["train"].with_transform(format_muse_textbooks).flatten()),
    (0.02, load_dataset("Ba2han/muse_textbooks", streaming=True)["train"].with_transform(format_muse_textbooks).flatten()),
    (0.01, load_dataset("amongglue/muse_textbooks", streaming=True)["train"].with_transform(format_muse_textbooks).flatten()),
    # domain specific art books
    (0.002, load_dataset("Nbardy/art-theory-textbooks", streaming=True)["train"].with_transform(format_all_you_need_books).flatten()),
    # Large Quality texbtbook sorce
    (0.05, load_dataset("SciPhi/textbooks-are-all-you-need-lite", streaming=True)["train"].with_transform(format_all_you_need_books).flatten()),
]


# Best!
# TODO: Make a dataset that is for a few epoch after the large pretraining in only grounded facts, and instruct models
# above is pretraining with large amounts of synthetic data that has high hallunicaiton rate as well as text data.
# textbooks_grounded = load_dataset("open-phi/textbooks_grounded", streaming=True)
#
# We need a new dataset that is instruct+grounded textbooks+chat


# Interleaving datasets with specified probabilities
dataset = interleave_datasets(
    datasets=all_datasets.map(lambda x: x[1]),
    probabilities=all.all_datasets.map(lambda x: x[0]),
    seed=42,  # Ensuring reproducibility
    stopping_strategy="all_exhausted",  # Continue until all datasets are exhausted
)

# For fast testing
dataset = dataset.select(range(10))

dataset = dataset.with_transform(tokenize_dataset)


def collate_data(batch):
    input_ids = torch.stack([item["input_ids"] for sublist in batch for item in sublist], dim=0)
    labels = torch.stack([item["labels"] for sublist in batch for item in sublist], dim=0)
    attention_masks = torch.stack([item["attention_mask"] for sublist in batch for item in sublist], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=6,
    collate_fn=collate_data,
)


# Modifying the collate function to handle chunks
def collate_data(batch):
    input_ids = torch.stack([item["input_ids"] for sublist in batch for item in sublist], dim=0)
    labels = torch.stack([item["labels"] for sublist in batch for item in sublist], dim=0)
    attention_masks = torch.stack([item["attention_mask"] for sublist in batch for item in sublist], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


###
# Validation Code
###

fake_abstract_test = "Abstract:\n This is a paper about novel machine learning techniques. We introduce a sparse MoE gate that is 4x more effeceint than"


def sample_validation(model, tokenizer, prompt, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


###
# Training Code
###

print("Starting training...")

wandb.init(project="tiny-moe")

step = 0

from tqdm import tqdm

# [Change] Updated the train loop to use the text field and added a tqdm progress bar
for epoch in range(args.epochs):
    model.train()

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        # This is debug code from
        # dtype = torch.bfloat16
        # in_debug = {name: tensor.dtype for name, tensor in batch.items()}
        # in_device_debug = {name: tensor.device for name, tensor in batch.items()}

        # raise Error("TODO: Dig into the inference code and figure out device erro")
        inputs = {name: tensor.to(device) for name, tensor in batch.items()}
        # inputs = {name: tensor for name, tensor in batch.items()}

        in_debug = {name: tensor.dtype for name, tensor in inputs.items()}
        in_device_debug = {name: tensor.device for name, tensor in inputs.items()}

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
        # print("Step:", step)
