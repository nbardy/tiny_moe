import torch
from transformers import AutoTokenizer
import time

import sys
import os


from train.configs import tiny_config, thin_and_wide, lightning_moe, lightning_moe_deep
from train.utils import print_params

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

MAX_TOKENS = 1024


# Function to run inference
def run_inference(model, tokenizer, prompts):
    model.eval()
    with torch.no_grad():
        start_time = time.time()

        with torch.autocast(device_type=device):
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(device)
            print(inputs)
            results = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
            # De

        end_time = time.time()

    return end_time - start_time, results


# Function to test a configuration
def test_config(config, config_name):
    print(f"Testing {config_name}...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")  # [Change] Corrected the tokenizer model name
    tokenizer.pad_token = tokenizer.eos_token  # [Change] Set the pad_token to eos_token for consistency
    model = TinyMoeForCausalLM(config)  # [Change] Initialized the model with the given config
    model.to(device, dtype=torch.bfloat16)  # [Change] Moved the model to the specified device and set dtype to bfloat16
    inference_time, results = run_inference(model, tokenizer, question_prompts)  # [Change] Run inference with the model, tokenizer, and prompts
    print("results")
    print(results)  # [Change] Print the inference results
    total_tokens = len(question_prompts) * MAX_TOKENS  # [Change] Calculate the total number of tokens
    # total_tokens should come from actually counting results.sequences

    percent_of_total = 150 / total_tokens  # [Change] Calculate the percentage of total tokens that 150 tokens represent
    time_for_150 = percent_of_total * inference_time  # [Change] Calculate the time for 150 tokens

    print("===" + config_name + "===")
    print_params(model, config)
    print(f"{config_name} took {inference_time:.2f} seconds for inference.\n")  # [Change] Print the inference time
    print(f"Total tokens: {total_tokens}\n")  # [Change] Print the total number of tokens
    print(f"Tokens per second: {total_tokens / inference_time:.2f}\n")  # [Change] Calculate and print tokens per second
    print(f"Time for 150 tokens from 5 batches: {time_for_150:.2f} seconds\n")  # [Change] Print the time for 150 tokens from 5 batches


# A test for gpt_2_medium as baseline
def test_gpt_2():
    # model_name = "openai/gpt-2-medium"  # [Change] Set the model name for GPT-2 medium
    model_name = "openai-community/gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # [Change] Initialize the tokenizer with the GPT-2 medium model
    from transformers import AutoModelForCausalLM  # [Change] Import the AutoModelForCausalLM class

    model = AutoModelForCausalLM.from_pretrained(model_name)  # [Change] Load the GPT-2 medium model

    model.to(device, dtype=torch.bfloat16)  # [Change] Move the model to the specified device and set dtype to bfloat16

    inference_time, results = run_inference(model, tokenizer, question_prompts)  # [Change] Run inference with the model, tokenizer, and prompts
    total_tokens = len(question_prompts) * MAX_TOKENS  # [Change] Calculate the total number of tokens
    percent_of_total = 150 / total_tokens  # [Change] Calculate the percentage of total tokens that 150 tokens represent
    time_for_150 = percent_of_total * inference_time  # [Change] Calculate the time for 150 tokens

    print("=== GPT 2 ====")
    print(f"{model_name} took {inference_time:.2f} seconds for inference.\n")  # [Change] Print the inference time for GPT-2 medium
    print(f"Total tokens: {total_tokens}\n")  # [Change] Print the total number of tokens
    print(f"Tokens per second: {total_tokens / inference_time:.2f}\n")  # [Change] Calculate and print tokens per second
    print(f"Time for 150 tokens from 5 batches: {time_for_150:.2f} seconds\n")  # [Change] Print the time for 150 tokens from 5 batches


# Test each configuration
# test_config(tiny_config, "Tiny Config")
# test_config(thin_and_wide, "Thin and Wide Config")
test_config(lightning_moe, "Lightning MoE")
test_config(lightning_moe_deep, "Lightning Moe Deep")
test_gpt_2()
