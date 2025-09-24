from safetensors.torch import load_file, save_file

# Paths
lokr_path = "/dheyo/varunika/output/lora_ab_base/lora_ab_base_LoRA_000004319.safetensors"
converted_path = "/dheyo/varunika/output/lora_ab_base/converted_lora_ab_base_LoRA_000004319.safetensors"

# Load the LoKR checkpoint
state_dict = load_file(lokr_path)

# Create a new state dict with mapped keys
new_state_dict = {}
for key, value in state_dict.items():
    if key == "emb_params":
        # Skip embedding parameters (handled separately in inference)
        continue
    # Map LoKR keys to diffusers-compatible LoRA keys
    if "transformer_transformer_blocks_" in key:
        new_key = key.replace("transformer_transformer_blocks_", "unet.")
        new_key = new_key.replace("lokr_A", "lora_A").replace("lokr_B", "lora_B")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Save the converted checkpoint
save_file(new_state_dict, converted_path)
print(f"Converted checkpoint saved to {converted_path}")