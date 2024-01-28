from transformers import AutoConfig
import torch

def load_config(config):
    base_config = AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-moe-16b-base", trust_remote_code=True
    )

    for key, value in config.items():
        setattr(base_config, key, value)
    
    return base_config

small_config = load_config({
    'n_routed_experts': 64,
    'n_shared_experts': 2,
    'num_experts_per_tok': 2,
    'hidden_size': 704,
    'intermediate_size': 2736, # 7168,
    'max_position_embeddings': 4096,
    'num_hidden_layers': 12,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,
    'torch_dtype': "bfloat16",
    # 'moe_intermediate_size': 4096,
    'moe_intermediate_size': 2200,
    'torch_dtype': torch.bfloat16,
    'attn_implementation': "flash_attention_2",
})

tiny_config = load_config({
    'n_routed_experts': 64,
    'n_shared_experts': 2,
    'num_experts_per_tok': 2,
    'hidden_size': 1024,
    'intermediate_size': 7168,
    'max_position_embeddings': 4096,
    'moe_intermediate_size': 1408,
    'num_hidden_layers': 14,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,
})

tiny_config_v2 = load_config({ # changes intermediate_size
    'n_routed_experts': 64,
    'n_shared_experts': 2,
    'num_experts_per_tok': 2,
    'hidden_size': 1024,
    'intermediate_size': 7168,
    'max_position_embeddings': 4096,
    'moe_intermediate_size': 1408,
    'num_hidden_layers': 14,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,
})

