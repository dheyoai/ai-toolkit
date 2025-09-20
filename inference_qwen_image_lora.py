from diffusers import DiffusionPipeline
import torch
from safetensors.torch import load_file # Removed save_file as it's not used here anymore
from transformers import AutoTokenizer 
from transformers import Qwen2_5_VLModel as TextEncoder
import json
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
import argparse
from pathlib import Path
import os
from typing import List, Optional
import time 
import re # Added for parsing step number from paths

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen-Image image generation script.")
    parser.add_argument(
        "--model_path", # "Qwen/Qwen-Image"
        type=str,
        required=True,
        help="Path to base model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path", # /dheyo/varunika/output/ab_qwenimage/tokenizer_0_ab_qwenimage__000004895
        type=str,
        default=None,
        help="Path to updated tokenizer (required for TI).",
    )
    parser.add_argument(
        "--text_encoder_path", # /dheyo/varunika/output/ab_qwenimage/text_encoder_0_ab_qwenimage__000004895
        type=str,
        default=None,
        help="Path to text encoder checkpoint (required for TI).",
    )
    parser.add_argument(
        "--embedding_path", # e.g., /dheyo/varunika/output/ab_qwenimage/[AB]_000004895.safetensors
        type=str,
        default=None,
        help="Path to Textual Inversion embedding .safetensors file (e.g., for [AB] token).",
    )
    
    # --- New arguments for AdaLoRA support ---
    parser.add_argument(
        "--network_type",
        type=str,
        default="lora",
        choices=["lora", "adalora"], # Add other types if needed, but these are the relevant ones for loading
        help="Type of network to load (lora or adalora).",
    )
    parser.add_argument(
        "--adalora_unet_adapter_path",
        type=str,
        default=None,
        help="Path to UNet AdaLoRA adapter directory (e.g., job_name_unet_adalora_adapter_XXXXX/).",
    )
    parser.add_argument(
        "--adalora_te_adapter_paths",
        type=str, # Comma-separated list of paths
        default=None,
        help="Comma-separated paths to Text Encoder AdaLoRA adapter directories (e.g., job_name_te0_adalora_adapter_XXXXX/,job_name_te1_adalora_adapter_XXXXX/).",
    )
    parser.add_argument(
        "--full_train_layers_path",
        type=str,
        default=None,
        help="Path to .safetensors file for full_train_in_out layers (if trained with AdaLoRA).",
    )
    # --- End new arguments for AdaLoRA support ---

    parser.add_argument(
        "--transformer_lora_path", # This will be for traditional LoRA, not AdaLoRA
        type=str,
        default=None,
        help="Path to transformer LoRA .safetensors checkpoint (for traditional LoRA).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output.png",
        help="Path to save output image."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        help="Path to prompts.txt for bulk generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="""
            aspect_ratios = {
                "1:1": (1024, 1024),
                "16:9": (1664, 928),
                "9:16": (928, 1664),
                "4:3": (1472, 1140),
                "3:4": (1140, 1472),
                "3:2": (1584, 1056),
                "2:3": (1056, 1584),
            }
        """,
    )
    # The --token_abstraction_json_path is explicitly removed as it's no longer used for prompt replacement
    # for TI embeddings. It can be kept for other purposes if necessary, but for now it's removed
    # from the arguments to avoid confusion.


    return parser.parse_args()


# The function convert_lora_weights_before_load is entirely removed
# as it was a workaround for a previous saving/loading issue and is
# not compatible with the new AdaLoRA saving mechanism or the standard
# diffusers LoRA loading.


def load_pipeline (args:argparse.Namespace):
    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    # --- Loading custom tokenizer and text encoder (required for TI) ---
    if args.tokenizer_path and args.text_encoder_path:
        print(f"Loading custom tokenizer from: {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Loading custom text encoder from: {args.text_encoder_path}")
        text_encoder = TextEncoder.from_pretrained(args.text_encoder_path,
                                                ignore_mismatched_sizes=True,
                                                torch_dtype=torch_dtype).to(device)
        text_encoder.resize_token_embeddings(len(tokenizer))
        pipe.tokenizer = tokenizer
        pipe.text_encoder = text_encoder 
    else:
        if args.embedding_path:
            print("Warning: Embedding path provided, but custom tokenizer/text_encoder paths are missing. "
                  "Textual Inversion embeddings may not load correctly or be recognized.")
    # --- End loading custom tokenizer and text encoder ---

    # --- Load Textual Inversion Embedding ---
    if args.embedding_path:
        print(f"Loading Textual Inversion embedding from: {args.embedding_path}")
        # The embedding_path likely points to a .safetensors file like '[AB]_000004895.safetensors'
        # This file contains the 'emb_params' which are the learned vectors for the TI token.
        try:
            embedding_sd = load_file(args.embedding_path)
            # Assuming the embedding_sd contains a single key (e.g., '[AB]') mapping to the embedding tensor.
            # And pipe.tokenizer and pipe.text_encoder are already set up.
            if hasattr(pipe, 'load_textual_inversion') and callable(pipe.load_textual_inversion):
                # Diffusers has a dedicated method for this, use it if available
                # It typically takes the folder or a specific file for the token
                # This assumes the TI embedding is saved in a compatible format for this method.
                pipe.load_textual_inversion(args.embedding_path)
                print(f"Successfully loaded Textual Inversion from {args.embedding_path} using pipe.load_textual_inversion.")
            else:
                # Manual injection for older/custom pipelines
                for token, embedding_tensor in embedding_sd.items():
                    # The token might be enclosed in '[]' or not, handle both
                    clean_token = token.strip('[]') if token.startswith('[') and token.endswith(']') else token
                    # Find the token ID in the (potentially resized) tokenizer
                    token_ids = pipe.tokenizer.encode(clean_token, add_special_tokens=False)
                    if len(token_ids) == 1:
                        token_id = token_ids[0]
                        if token_id != pipe.tokenizer.unk_token_id:
                            # Assuming embedding_tensor is directly the learned vector
                            if embedding_tensor.dim() == 1: # If it's just the vector
                                pipe.text_encoder.embeddings.word_embeddings.weight.data[token_id] = embedding_tensor.to(torch_dtype).to(device)
                                print(f"Manually injected Textual Inversion embedding for '{clean_token}' (ID: {token_id}).")
                            elif embedding_tensor.dim() == 2 and embedding_tensor.shape[0] == 1: # If it's a 1-item batch
                                pipe.text_encoder.embeddings.word_embeddings.weight.data[token_id] = embedding_tensor.squeeze(0).to(torch_dtype).to(device)
                                print(f"Manually injected Textual Inversion embedding for '{clean_token}' (ID: {token_id}).")
                            else:
                                print(f"Warning: TI embedding for '{clean_token}' has unexpected shape {embedding_tensor.shape}. Skipping manual injection.")
                        else:
                            print(f"Warning: TI token '{clean_token}' not found in tokenizer vocabulary. Skipping manual injection.")
                    else:
                        print(f"Warning: TI token '{clean_token}' maps to multiple or zero token IDs. Skipping manual injection.")
        except Exception as e:
            print(f"Error loading Textual Inversion embedding from {args.embedding_path}: {e}")
    # --- End load Textual Inversion Embedding ---

    # --- Load Network Weights (AdaLoRA or traditional LoRA) ---
    if args.network_type.lower() == "adalora":
        print("Loading AdaLoRA network...")
        
        # Load UNet AdaLoRA adapter
        if args.adalora_unet_adapter_path and os.path.isdir(args.adalora_unet_adapter_path):
            print(f"Loading UNet AdaLoRA adapter from: {args.adalora_unet_adapter_path}")
            # The adapter_name here is internal to diffusers/peft for managing multiple adapters.
            # You might need to derive it from your training setup.
            # For simplicity, we'll use a generic name, but for multiple adapters, ensure uniqueness.
            pipe.load_lora_weights(args.adalora_unet_adapter_path, adapter_name="adalora_unet")
        else:
            if args.adalora_unet_adapter_path:
                print(f"Warning: UNet AdaLoRA adapter path {args.adalora_unet_adapter_path} not found or is not a directory. Skipping.")

        # Load Text Encoder AdaLoRA adapters
        if args.adalora_te_adapter_paths:
            te_adapter_paths = [p.strip() for p in args.adalora_te_adapter_paths.split(',') if p.strip()]
            for i, te_path in enumerate(te_adapter_paths):
                if os.path.isdir(te_path):
                    print(f"Loading Text Encoder {i} AdaLoRA adapter from: {te_path}")
                    pipe.load_lora_weights(te_path, adapter_name=f"adalora_te_{i}")
                else:
                    print(f"Warning: Text Encoder {i} AdaLoRA adapter path {te_path} not found or is not a directory. Skipping.")

        # Load full_train_in_out layers if available
        if args.full_train_layers_path and os.path.exists(args.full_train_layers_path):
            print(f"Loading full_train_in_out layers from: {args.full_train_layers_path}")
            full_train_sd = load_file(args.full_train_layers_path)
            
            # Apply state_dict to the corresponding base modules in the pipeline
            # The keys might be prefixed (e.g., 'unet_conv_in.weight')
            if hasattr(pipe.unet, 'conv_in') and any(k.startswith('unet_conv_in.') for k in full_train_sd.keys()):
                pipe.unet.conv_in.load_state_dict({k.replace('unet_conv_in.', ''): v for k, v in full_train_sd.items() if k.startswith('unet_conv_in.')})
                print("Loaded UNet conv_in layers.")
            if hasattr(pipe.unet, 'conv_out') and any(k.startswith('unet_conv_out.') for k in full_train_sd.keys()):
                pipe.unet.conv_out.load_state_dict({k.replace('unet_conv_out.', ''): v for k, v in full_train_sd.items() if k.startswith('unet_conv_out.')})
                print("Loaded UNet conv_out layers.")
            # For Qwen-Image's transformer blocks, it might have pos_embed or proj_out directly
            if hasattr(pipe.unet, 'pos_embed') and any(k.startswith('transformer_pos_embed.') for k in full_train_sd.keys()):
                pipe.unet.pos_embed.load_state_dict({k.replace('transformer_pos_embed.', ''): v for k, v in full_train_sd.items() if k.startswith('transformer_pos_embed.')})
                print("Loaded Transformer pos_embed layers.")
            if hasattr(pipe.unet, 'proj_out') and any(k.startswith('transformer_proj_out.') for k in full_train_sd.keys()):
                pipe.unet.proj_out.load_state_dict({k.replace('transformer_proj_out.', ''): v for k, v in full_train_sd.items() if k.startswith('transformer_proj_out.')})
                print("Loaded Transformer proj_out layers.")
            
    elif args.network_type.lower() == "lora":
        if args.transformer_lora_path and os.path.exists(args.transformer_lora_path):
            print(f"Loading traditional LoRA from: {args.transformer_lora_path}")
            lora_dir = os.path.dirname(args.transformer_lora_path)
            lora_weight_name = os.path.basename(args.transformer_lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=lora_weight_name)
        else:
            if args.transformer_lora_path:
                print(f"Warning: Traditional LoRA path {args.transformer_lora_path} not found. Skipping.")
    else:
        print(f"Warning: Network type '{args.network_type}' specified, but no corresponding weights path provided or found. Running without network.")

    # --- End Load Network Weights ---

    return pipe

def main (args:argparse.Namespace, prompts: List) -> None:
    # Ensure custom tokenizer and text_encoder paths are provided if loading embeddings
    if args.embedding_path and (not args.tokenizer_path or not args.text_encoder_path):
        raise ValueError("If --embedding_path is provided, --tokenizer_path and --text_encoder_path must also be provided.")

    pipe = load_pipeline(args)
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition." # for english prompt
    }

    # The token abstraction JSON path and prompt replacement logic is removed.
    # Textual Inversion tokens (like [AB]) should be directly handled by the
    # loaded tokenizer and text encoder.
    # if args.token_abstraction_json_path:
    #     with open(args.token_abstraction_json_path, "r") as file:
    #         representation_tokens = json.load(file)
    #     special_tokens = list(representation_tokens.keys())


    aspect_ratios = {
        "1:1": (1024, 1024),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios[args.aspect_ratio]

    for idx, prompt in enumerate(prompts):
        # Removed the prompt replacement logic.
        # The [AB] token should be present in the prompt and recognized by the loaded tokenizer.
        # if args.token_abstraction_json_path:
        #     for special_token in special_tokens:
        #         prompt = prompt.replace(special_token, representation_tokens[special_token][0].replace(" ", ''))
        
        # Ensure trigger token "ab" is always included at the start if it isn't already.
        # This is for a general trigger or domain word, distinct from the specific TI token [AB].
        if not prompt.strip().startswith("ab "): 
            prompt = "ab " + prompt
            print("Prepending 'ab ' to prompt (general style/domain trigger).")

        print(f"Using prompt: {prompt}")

        images = pipe(
            num_images_per_prompt=args.num_images_per_prompt,
            prompt=prompt + " " + positive_magic["en"], # Added a space before positive_magic for better parsing
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed)
        ).images

        os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
        timestamp = str(time.strftime("%d-%m-%y_%H-%M-%S"))

        for image_id, image in enumerate(images):
            # Example output filename: inferenced_images/ab_woman_leather_jacket_0_0_2025-09-20_10-00-00.png
            file_path = f"{args.output_image_path.replace('.png', '')}_{idx}_{image_id}_{timestamp}.png"
            image.save(file_path)
            print(f"✅ Saved {file_path}")


if __name__ == '__main__':
    args = parse_args()
    if not args.instruction and not args.prompts_path:
        raise ValueError("Either --instruction or --prompts_path has to be specified, both are None")

    if args.prompts_path:
        prompts_path = Path(args.prompts_path)
        prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    else:
        prompts = [args.instruction]
    main(args, prompts)