from datasets import interleave_datasets
import random
import logging


from datasets import load_dataset
from transformers import (
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

import random
import pyarrow
import pandas as pd
import json

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

parser.add_argument("--max_chunk_size", type=int, default=1024, help="Max chunk size")

# Parse the arguments
args = parser.parse_args()

MAX_CHUNK_SIZE = args.max_chunk_size

# Specify the new configuration

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
dtype = torch.bfloat16

# Initialize the model with the new configuration
model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
model.to(device, dtype=dtype)


# Choose the optimizer based on the parsed argument
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001, fused=True)
    lr_scheduler_type = "cosine"
    # Cosine annealing
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs
    )  # [Change] Added num_training_steps to lr_scheduler
elif args.optimizer == "prodigy":
    optimizer = Prodigy(model.parameters(), lr=1.0, weight_decay=0.0001)
    lr_scheduler = get_constant_schedule(optimizer)
    print("Learning rate is ignored for Prodigy optimizer")

print_params(model, config)

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


# Setup logging for data warnings
logging.basicConfig(filename="data_warn.log", level=logging.WARNING, format="%(asctime)s:%(levelname)s:%(message)s")


def format_item_fn(header_columns, prefix="", chunk_size=MAX_CHUNK_SIZE, base_key="markdown"):
    def formatter(item):
        try:
            # Check if base_key exists in item and is not None or empty, else log warning and skip processing
            if base_key not in item or not item[base_key]:  # [Change] Skip processing if base_key is missing, None, or empty
                logging.warning(f"Missing or empty base_text for item: {item}")  # [Change] Log warning for missing or empty base_text
                return []

            base_text = str(item[base_key])  # Convert to string to ensure compatibility with split method
            # Split base_text into chunks, handling cases where base_text might be empty
            markdown_chunks = (
                [" ".join(base_text.split()[i : i + chunk_size]) for i in range(0, len(base_text.split()), chunk_size)] if base_text else []
            )  # [Change] Handle empty base_text case

            all_chunks = []

            for chunk in markdown_chunks:
                # Safely access item keys for header_text, handling missing keys gracefully
                header_text = " ".join(f"{key}: {item.get(key, 'N/A')}" for key in header_columns)  # [Change] Use item.get to handle missing keys
                formatted_text = f"{prefix} {header_text} Excerpt: {chunk}"
                all_chunks.append({"text": formatted_text})

            # Do one chunk with just prefix, ensuring there's at least one chunk to choose from
            if prefix != "" and markdown_chunks:  # [Change] Check markdown_chunks is not empty before choosing a random chunk
                random_chunk = random.choice(markdown_chunks)
                formatted_text = f"{prefix} {random_chunk}"
                all_chunks.append({"text": formatted_text})

            # Forma to pyarrow table
            import pyarrow
            import pandas as pd

            table = pyarrow.Table.from_pandas(pd.DataFrame(all_chunks))

            return table

        except Exception as e:
            # Log exception details
            logging.exception(f"Exception occurred while formatting item: {item}")  # [Change] Log exception details
            return []  # Return empty list in case of exception

    return formatter


def format_chat(item, column_name="conversations", from_key="from", value_key="value", prefix="", chunk_size=MAX_CHUNK_SIZE):
    """
    Formats chat logs into a string or text chat log format based on a random mode.
    Args:
        item (dict): The item containing the chat logs.
        column_name (str): The key in the item dict that contains the chat logs.
        from_key (str): The key in each chat log dict that identifies the sender.
        value_key (str): The key in each chat log dict that contains the message text.
    Returns:
        pyarrow.Table: A table containing the formatted chat logs.
    """

    # Ensure the column exists in the item
    if column_name not in item or not item[column_name]:
        logging.warning(f"Missing or empty {column_name} for item: {item}")
        return []

    chat_logs = item[column_name]
    mode = random.choice(["string", "chat_log"])

    all_chunks = []

    if mode == "string":
        # Dump json
        json_str = json.dumps(chat_logs)
        # Split into chunk size chunks and label with "{prefix} ""
        chunk_size = 512  # Define chunk size for splitting the json string
        chat_string = ""
        for i in range(0, len(json_str), chunk_size):
            chunk = json_str[i : i + chunk_size]
            if prefix:
                chunk = f"{prefix} {chunk}"
            chat_string += chunk + " " if i + chunk_size < len(json_str) else chunk
        all_chunks.append({"text": chat_string})
    else:
        # Format each chat message into a 'from: message' format
        chat_formatted = "\n".join([f"{log[from_key]}: {log[value_key]}" for log in chat_logs if from_key in log and value_key in log])
        all_chunks.append({"text": chat_formatted})

    # Convert to pyarrow table
    table = pyarrow.Table.from_pandas(pd.DataFrame(all_chunks))

    return table


format_science_papers = format_item_fn(["absract", "article"], prefix="Science paper", base_key="article")
format_wiki = format_item_fn(["title", "categories"], prefix="Wikipedia", base_key="markdown")
format_mathpile_chunks = format_item_fn([], prefix="Math Pile", base_key="text")
format_pile_chunks = format_item_fn([], prefix="Web Text", base_key="text")
format_grounded_textbooks = format_item_fn(["topic", "model", "concepts", "outline"], prefix="Grounded Synthetic Textbook")
format_synthetic_textbooks = format_item_fn(["model", "topic", "concepts", "outline"], prefix="Synthetic Textbooks")
format_fake_textbooks = format_item_fn(["model", "topic", "outline", "field", "subfield"], prefix="Fake Textbooks(Some Grounded)")
format_orca_textbooks = format_item_fn([], base_key="textbook", prefix="Tiny Orca Textbooks")
format_strange_textbooks = format_item_fn([], prefix="Tiny Strange Textbooks", base_key="text")
format_math_textbooks = format_item_fn([], prefix="Tiny Math Textbooks", base_key="text")
format_code_textbooks = format_item_fn([], prefix="Tiny Code Textbooks", base_key="text")
format_tiny_textbooks = format_item_fn([], prefix="Tiny Textbooks", base_key="text")
format_muse_textbooks = format_item_fn([], prefix="Muse Textbooks", base_key="text")
format_all_you_need_books = format_item_fn([], prefix="Synthetic Code Textbooks", base_key="text")
format_anthropic_chat = format_item_fn([], prefix="Safe Chat(Chosen)", base_key="chosen")
format_anthropic_chat_rejected = format_item_fn([], prefix="Safe Chat(Rejected)", base_key="rejected")

format_auto_math_text = format_item_fn([], prefix="Math Papers", base_key="text")
format_hq_math_text = format_item_fn([], prefix="Quality Math Papers", base_key="text")
format_auto_math_code = format_item_fn([], prefix="Math Code", base_key="text")


def map_batch(dataset, format_function, num_process=16):
    """
    Wrapper function to map a dataset with the specified format function in batches and parallel processing.
    """
    return dataset.map(format_function, batched=True, num_proc=num_process)


def load_data(dataset_name, subset_name, transform_function):
    """
    Loads a dataset with streaming enabled and applies a transform function in batched mode.

    Args:
        dataset_name (str): The name of the dataset to load.
        subset_name (str): The subset name of the dataset.
        transform_function (function): The transform function to apply to the dataset.

    Returns:
        Dataset: The loaded and transformed dataset.
    """
    print(f"Loading {dataset_name}/{subset_name}")
    dataset = load_dataset(dataset_name, subset_name, streaming=True)
    transformed_dataset = dataset["train"].map(transform_function, batched=True)
    return transformed_dataset


all_datasets = [
    # 30% web text
    (0.15, load_data("JeanKaddour/minipile", "train", format_pile_chunks)),  # Restoring the comment about web text
    # 34% papers on xaxriv
    (0.14, load_data("GAIR/MathPile_Commercial", "train", format_mathpile_chunks)),  # Adding back the comment about papers
    (0.06, load_data("scientific_papers", "arxiv", format_science_papers)),  # Continuing the comment about papers
    (
        0.02,
        load_data("math-ai/AutoMathText", "code-python-0.80-to-1.00", format_auto_math_text),
    ),  # Keeping the comment about different versions commented out
    (0.04, load_data("math-ai/AutoMathText", "web-0.80-to-1.00", format_hq_math_text)),
    (0.04, load_data("math-ai/AutoMathText", "arxiv-0.80-to-1.00", format_hq_math_text)),
    # 5% wiki
    (0.05, load_data("euirim/goodwiki", "train", format_wiki)),  # Adding the comment about wiki
    # Rest is synthetic textbooks
    (
        0.05,
        load_data("open-phi/textbooks_grounded", "train", format_grounded_textbooks),
    ),  # Comment about synthetic textbooks restored
    (0.002, load_data("Nbardy/art-theory-textbooks", "train", format_all_you_need_books)),
    (0.05, load_data("SciPhi/textbooks-are-all-you-need-lite", "train", format_all_you_need_books)),
    (0.04, load_data("Anthropic/hh-rlhf", "train", format_anthropic_chat)),
    (0.04, load_data("Anthropic/hh-rlhf", "train", format_anthropic_chat_rejected)),
]


# Best!
# TODO: Make a dataset that is for a few epoch after the large pretraining in only grounded facts, and instruct models
# above is pretraining with large amounts of synthetic data that has high hallunicaiton rate as well as text data.
# textbooks_grounded = load_dataset("open-phi/textbooks_grounded", streaming=True)
#
# We need a new dataset that is instruct+grounded textbooks+chat


# Interleaving datasets with specified probabilities
dataset = interleave_datasets(
    datasets=list(map(lambda x: x[1], all_datasets)),
    probabilities=list(map(lambda x: x[0], all_datasets)),
    seed=42,  # Ensuring reproducibility
    stopping_strategy="all_exhausted",  # Continue until all datasets are exhausted
)

# For fast testing
ataset = dataset.map(tokenize_dataset, batched=True, num_proc=12)


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
        inputs = {name: tensor.to(device) for name, tensor in batch.items()}

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
