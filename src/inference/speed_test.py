import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.configs import small_config, tiny_config, thin_and_wide
import time

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
            model.generate(inputs.input_ids, max_length=30)
        end_time = time.time()
    return end_time - start_time


# Function to test a configuration
def test_config(config, config_name):
    print(f"Testing {config_name}...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    inference_time = run_inference(model, tokenizer, question_prompts)
    print(f"{config_name} took {inference_time:.2f} seconds for inference.\n")


# Test each configuration
test_config(small_config, "Small Config")
test_config(tiny_config, "Tiny Config")
test_config(thin_and_wide, "Thin and Wide Config")
