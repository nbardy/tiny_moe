from transformers import AutoConfig
import torch

from models.configuration_tinymoe import TinyMoeConfig


def load_config(config):
    for key, value in config.items():
        setattr(base_config, key, value)

    return base_config


base_config = TinyMoeConfig(
    {
        # special Tokens
        "bos_token_id": 1,
        "eos_token_id": 2,
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
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "attention_bias": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "vocab_size": 32000,
        "output_router_logits": False,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "router_aux_loss_coef": 0.02,
        "hidden_act": "silu",
        "first_k_dense_replace": 1,
        "scoring_func": "softmax",
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

# The final architecture is a blend of wide Deepseek style moe MLP blocks
# and sliding window attention
#
# We program different sized attention layers, biasing toward later
# laters with more params.
#
# We use cheaper sliding attention at the start and then switch to full attention
# at the end.
# We use a different per layer config.
#
# Less parameter count in the first layers
# We start with small sliding window attention and increase the resolution toward the final layers
lightning_moe = load_config(
    {
        # MoeMLP layers
        "n_routed_experts": 32,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        # "hidden_size": 4096,
        "hidden_size": 1024,
        "moe_intermediate_size": 512,
        "intermediate_size": 1024,
        # Attention layers
        "num_hidden_layers": 20,
        "max_position_embeddings": 4096,
        "window_sizes": [64, 4096, 64, 64, 64, 1024, 64, 4096],
        "num_attention_heads": [8, 8, 16, 16, 64, 64, 64, 64, 64],
        "num_key_value_heads": [8, 8, 2, 2, 8, 8, 8, 8, 64],
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "_attn_implementation": "flash_attention_2",
        "vocab_size": 32000,
    }
)
