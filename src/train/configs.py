from transformers import AutoConfig
import torch


def load_config(config):
    base_config = AutoConfig.from_pretrained("deepseek-ai/deepseek-moe-16b-base", trust_remote_code=True)

    for key, value in config.items():
        setattr(base_config, key, value)

    return base_config


small_config = load_config(
    {
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        "hidden_size": 704,
        "intermediate_size": 2736,  # 7168,
        "max_position_embeddings": 4096,
        "num_hidden_layers": 12,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
        # 'moe_intermediate_size': 4096,
        "moe_intermediate_size": 2200,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "vocab_size": 32000,
    }
)

tiny_config = load_config(
    {
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        "hidden_size": 1056,
        "intermediate_size": 4096,
        "max_position_embeddings": 4096,
        "moe_intermediate_size": 1408,
        "num_hidden_layers": 12,
        "num_attention_heads": 24,
        "num_key_value_heads": 6,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "vocab_size": 32000,
    }
)

thin_and_wide = load_config(
    {
        "n_routed_experts": 128,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 128,
        "intermediate_size": 1024,
        "max_position_embeddings": 4096,
        "moe_intermediate_size": 512,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "vocab_size": 32000,
    }
)

# We use cheaper sliding attention at the start and then switch to full attention
# at the end.
# We use a different per layer config.
#
# Less parameter count in the first layers
# We start with small sliding window attention and increase the resolution toward the final layers
thin_and_wide_sliding_attention_pyramid = load_config(
    {
        # MoeMLP layers
        "n_routed_experts": 32,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        "hidden_size": 128,
        "moe_intermediate_size": 512,
        "intermediate_size": 1024,
        # Attention layers
        "num_hidden_layers": 16,
        "max_position_embeddings": 4096,
        "window_sizes": [64, 4096, 64, 256, 1024, 64, 4096],
        "num_attention_heads": [8, 64, 64, 64, 64, 64, 64, 256],
        "num_key_value_heads": [8, 8, 8, 8, 8, 8, 64, 64],
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "vocab_size": 32000,
    }
)
