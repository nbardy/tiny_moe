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


# Function to run inference
def run_inference(model, tokenizer, prompts):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model.generate(**inputs, max_length=1024)
        end_time = time.time()
    return end_time - start_time


# Function to test a configuration
def test_config(config, config_name):
    print(f"Testing {config_name}...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = TinyMoeForCausalLM(config)
    model.to(device, dtype=torch.bfloat16)
    inference_time = run_inference(model, tokenizer, question_prompts)
    print(f"{config_name} took {inference_time:.2f} seconds for inference.\n")


# Test each configuration
# test_config(small_config, "Small Config")
# test_config(tiny_config, "Tiny Config")
# test_config(thin_and_wide, "Thin and Wide Config")
test_config(lightning_moe, "Lightning MoE")
