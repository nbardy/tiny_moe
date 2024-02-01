from transformers import AutoTokenizer, AutoModelForCausalLM
from configs import tiny_config
import torch

# Load the configuration
config = tiny_config

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")

# Initialize the model with the configuration
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the prompt
prompt = "Abstract:\n This is a paper about novel machine learning techniques. We introduce a sparse MoE gate that is 4x more effeceint than"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
max_length = 300  # you can adjust this value
generate_ids = model.generate(inputs.input_ids, max_length=max_length)

# Decode the generated text
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(generated_text)
