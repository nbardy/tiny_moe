from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule
from train_dataloader import getEffecientPileV1

from configs import small_config, tiny_config

from utils import print_params
from prodigyopt import Prodigy
import torch
import argparse
import wandb

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

    dataset = getEffecientPileV1(shuffle=True, tokenizer)

    for batch in tqdm(dataset, desc=f"Epoch {epoch+1}"):
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
