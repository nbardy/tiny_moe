import torch
from transformers import AutoTokenizer
import time

import sys
import os

from train.configs import tiny_config, thin_and_wide, lightning_moe

from models.modeling_tinymoe import TinyMoeForCausalLM


# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the batch of question prompts
question_prompts = [
    "What is the capital of France?",
    "Who wrote the book '1984'?",
    "What is the chemical symbol for water?",
    "How far is the Moon from Earth?",
    "What is the speed of light?",
]

TOKENS = 1024


# Function to run inference
def run_inference(model, tokenizer, prompts):
    model.eval()
    with torch.no_grad():
        start_time = time.time()

        with torch.autocast(device_type=device):
            inputs = tokenizer(prompts, return_tensors="pt").to(device)
            results = model.generate(**inputs, max_new_tokens=TOKENS)

        end_time = time.time()

    return end_time - start_time, results


# Function to test a configuration
def test_config(config, config_name):
    print(f"Testing {config_name}...")
    # With autocast to bf16
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = TinyMoeForCausalLM(config)
    model.to(device, dtype=torch.bfloat16)
    inference_time, results = run_inference(model, tokenizer, question_prompts)
    print(results)
    total_tokens = len(question_prompts) * TOKENS
    print(f"{config_name} took {inference_time:.2f} seconds for inference.\n")
    print(f"Total tokens: {total_tokens}\n")
    # Tokens per second
    print(f"Tokens per second: {total_tokens / inference_time:.2f}\n")
    # Time for 150 tokens from 5 batchs = 5 * 150 / tokens per second
    percent_of_total = 150 / total_tokens
    time_for_150 = percent_of_total * inference_time
    print(f"Time for 150 tokens from 5 batchs: {time_for_150:.2f} seconds\n")


# Test each configuration
# test_config(small_config, "Small Config")
# test_config(tiny_config, "Tiny Config")
# test_config(thin_and_wide, "Thin and Wide Config")
test_config(lightning_moe, "Lightning MoE")
