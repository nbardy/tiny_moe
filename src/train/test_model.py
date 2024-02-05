from transformers import AutoTokenizer, AutoModelForCausalLM
from configs import tiny_config, thin_and_wide
import torch
import time


# Function to test configurations and measure inference speed
def test_config_and_inference_speed(config_name, config, prompt, max_length=300):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")

    # Initialize the model with the configuration
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attn_2")

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # Move model to GPU if available
    device = torch.device("cuda")
    model.to(device)

    print(f"Testing configuration: {config_name}")
    for batch_size in range(1, 2):  # Test batch sizes from 1 to 10
        # Repeat the prompt to simulate batch processing
        batched_prompt = [prompt] * batch_size
        # Tokenize the batched prompt
        inputs = tokenizer(batched_prompt, return_tensors="pt", padding=True).to(device)

        # Measure inference time
        start_time = time.time()
        generate_ids = model.generate(**inputs, max_length=max_length)
        end_time = time.time()

        # Decode the generated text for the first item in the batch
        generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

        # Calculate tokens generated and tokens per second
        tokens_generated = len(generate_ids[0])
        inference_time = end_time - start_time
        tokens_per_second = tokens_generated / inference_time

        # Log the time, size, tokens generated, and tokens per second
        print(
            f"Batch size: {batch_size}, Time: {inference_time:.2f}s, Generated text size: {len(generated_text)} characters, Tokens: {tokens_generated}, Tokens/s: {tokens_per_second:.2f}"
        )


# Define the prompt
prompt = "Abstract:\n This is a paper about novel machine learning techniques. We introduce a sparse MoE gate that is 4x more efficient than"

# Test both configurations
test_config_and_inference_speed("tiny_config", tiny_config, prompt)
test_config_and_inference_speed("thin_and_wide", thin_and_wide, prompt)
