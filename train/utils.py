def print_params(model, config):
    print("----- Model Summary ------")
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params}")

    # Print in human readable XM where M standard for M and X is rounded
    # Also print in XB where X is B
    # then print total / experts_count(66) as Per expert count

    # Then print total active params = per_expert_params * 4
    # Calculate and print the total parameters in human readable format
    total_params_millions = total_params / 1e6
    total_params_billions = total_params / 1e9
    print(f"Total parameters: {total_params_millions:.2f}M ({total_params_billions:.2f}B)")

    # Calculate parameters per expert and print
    experts_count = config.n_routed_experts + config.n_shared_experts
    per_expert_params = total_params / experts_count
    per_expert_params_billions = per_expert_params / 1e6  # [Change] Added parameter count in billions
    print(f"Parameters per expert: {per_expert_params:.2f} ({per_expert_params_billions:.2f}M)")  # [Change] Added parameter count in billions

    # Calculate and print total active parameters
    total_active_params = per_expert_params * 4
    total_active_params_billions = total_active_params / 1e6  # [Change] Added active parameter count in billions
    print(f"Total active parameters: {total_active_params:.2f} ({total_active_params_billions:.2f}M)")  # [Change] Added active parameter count in billions
    print("----------------")

