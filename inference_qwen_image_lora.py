import torch
from diffusers import QwenImagePipeline
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from safetensors.torch import load_file
import json
import argparse
from pathlib import Path
import os
import time
import re
from typing import List

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen-Image image generation with safetensors LoRA and Textual Inversion.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base Qwen-Image model checkpoint (e.g., 'Qwen/Qwen-Image').",
    )
    parser.add_argument(
        "--lora_type",
        type=str,
        default="lora",
        choices=["lora"],
        help="Type of LoRA to load: 'lora' for safetensors weights.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA .safetensors file.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to updated tokenizer.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to text encoder checkpoint.",
    )
    parser.add_argument(
        "--token_abstraction_json_path",
        type=str,
        default=None,
        help="Path to token abstraction dict (e.g., 'tokens.json').",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
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
        help="Base path for saving output images."
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
        help="Path to a .txt file containing prompts for bulk generation."
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="Supported aspect ratios: '1:1': (1024, 1024), '16:9': (1664, 928), '9:16': (928, 1664), "
             "'4:3': (1472, 1140), '3:4': (1140, 1472), '3:2': (1584, 1056), '2:3': (1056, 1584).",
    )
    return parser.parse_args()

def remap_qwen_lora_keys_for_pipeline(state_dict: dict) -> dict:
    """
    Remaps LoRA keys to match QwenImagePipeline's transformer module expectations.
    """
    remapped_sd = {}
    print("Original state_dict keys:")
    for key in state_dict.keys():
        print(key)
    
    LORA_SUFFIX_REGEX = re.compile(r"(\.lora_down|\.lora_up)\.(weight)$")
    
    for original_key, value in state_dict.items():
        print(f"\nProcessing key: {original_key}")
        match = LORA_SUFFIX_REGEX.search(original_key)
        if not match:
            print(f"Skipping non-LoRA key: {original_key}")
            continue
        
        base_module_name = original_key[:match.start()]
        lora_suffix = original_key[match.start():]
        print(f"Base: {base_module_name}, Suffix: {lora_suffix}")
        
        # Remove 'transformer.' prefix to match 'transformer_blocks.<block_id>.<module>'
        if base_module_name.startswith("transformer.transformer_blocks."):
            base_module_name = base_module_name.replace("transformer.transformer_blocks.", "transformer_blocks.")
            print(f"Remapped prefix to: {base_module_name}")
        
        final_remapped_key = base_module_name + lora_suffix
        print(f"Final remapped key: {final_remapped_key}")
        remapped_sd[final_remapped_key] = value
    
    print("\nRemapped state_dict keys:")
    for key in remapped_sd.keys():
        print(key)
    
    if not remapped_sd:
        print("Warning: No LoRA modules remained after remapping. The LoRA checkpoint might be incompatible or empty.")
    
    return remapped_sd

def load_pipeline(args: argparse.Namespace):
    """Load the QwenImagePipeline with safetensors LoRA and updated tokenizer/text encoder."""
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading base pipeline from {args.model_path} on {device} with dtype {torch_dtype}...")
    try:
        pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"Error loading base pipeline: {e}")
        raise
    
    lora_loaded = False
    if args.lora_type == "lora" and args.lora_path and Path(args.lora_path).is_file():
        print(f"Loading safetensors LoRA weights from {args.lora_path}")
        try:
            state_dict = load_file(args.lora_path, device="cpu")
            if "emb_params" in state_dict:
                print("Found Textual Inversion embeddings in LoRA checkpoint; skipping for transformer LoRA.")
                state_dict.pop("emb_params")
            
            remapped_lora_sd = remap_qwen_lora_keys_for_pipeline(state_dict)
            if not remapped_lora_sd:
                print("Warning: No LoRA modules remained after remapping. Skipping LoRA loading.")
            else:
                transformer_state_dict = pipe.transformer.state_dict()
                transformer_state_dict.update(remapped_lora_sd)
                pipe.transformer.load_state_dict(transformer_state_dict, strict=False)
                lora_loaded = True
                print("Safetensors LoRA weights loaded successfully into transformer.")
        except Exception as e:
            print(f"Error loading safetensors LoRA weights: {e}")
    
    if not lora_loaded:
        print("No LoRA weights were loaded. Proceeding with base model only.")
    
    if args.tokenizer_path and args.text_encoder_path:
        print(f"Loading updated tokenizer from {args.tokenizer_path} and text encoder from {args.text_encoder_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, torch_dtype=torch_dtype)
            text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                args.text_encoder_path, torch_dtype=torch_dtype, ignore_mismatched_sizes=True
            ).to(device)
            text_encoder.resize_token_embeddings(len(tokenizer))
            pipe.tokenizer = tokenizer
            pipe.text_encoder = text_encoder
            print("Updated tokenizer and text encoder loaded and resized.")
        except Exception as e:
            print(f"Error loading tokenizer or text encoder: {e}")
    
    # Optimize for GPU memory
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    
    return pipe

def main(args: argparse.Namespace, prompts: List[str]) -> None:
    """Main function to generate images with the QwenImagePipeline."""
    pipe = load_pipeline(args)
    
    positive_magic = {"en": "Ultra HD, 4K, cinematic composition."}
    aspect_ratios = {
        "1:1": (1024, 1024), "16:9": (1664, 928), "9:16": (928, 1664),
        "4:3": (1472, 1140), "3:4": (1140, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584),
    }
    
    width, height = aspect_ratios.get(args.aspect_ratio, (1024, 1024))
    print(f"Generating images with aspect ratio {args.aspect_ratio} ({width}x{height}).")
    
    representation_tokens = {}
    selected_token = "[AB]_0"  # Default to first trained token
    if args.token_abstraction_json_path and Path(args.token_abstraction_json_path).is_file():
        with open(args.token_abstraction_json_path, "r") as file:
            representation_tokens = json.load(file)
        print(f"Loaded {len(representation_tokens)} Textual Inversion tokens for prompt abstraction.")
        
        # Parse tokens.json to extract [AB]_0 to [AB]_31
        if "[AB]" in representation_tokens:
            concepts = representation_tokens["[AB]"]
            if isinstance(concepts, list) and len(concepts) > 0:
                # Split the string into individual tokens (e.g., [AB]_0, [AB]_1, ...)
                token_list = concepts[0].split()
                if token_list and all(re.match(r"\[AB\]_\d+", t) for t in token_list):
                    print(f"Found trained tokens: {token_list}")
                    selected_token = token_list[0]  # Use [AB]_0 as default
                else:
                    print("Warning: tokens.json format unexpected. Using default token [AB]_0.")
    
    for idx, prompt_orig in enumerate(prompts):
        # Replace [AB] with selected_token (e.g., [AB]_0)
        prompt_to_generate = prompt_orig.replace("[AB]", selected_token)
        
        print(f"Prompt after TI token replacement: '{prompt_to_generate}'")
        print(f"--- Generating for original prompt: '{prompt_orig}' ---")
        
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        
        try:
            images = pipe(
                num_images_per_prompt=args.num_images_per_prompt,
                prompt=prompt_to_generate + positive_magic["en"],
                negative_prompt=args.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=generator
            ).images
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
        
        output_dir = Path(args.output_image_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%d-%m-%y_%H-%M-%S")
        clean_prompt_for_filename = re.sub(r'[^a-zA-Z0-9_ -]', '', prompt_orig[:50]).strip() or f"prompt_{idx}"
        
        for image_id, image in enumerate(images):
            file_path = output_dir / f"{clean_prompt_for_filename}_{idx}_{timestamp}.png"
            image.save(file_path)
            print(f"Saved {file_path}")

if __name__ == '__main__':
    args = parse_args()
    if not args.instruction and not args.prompts_path:
        raise ValueError("Either --instruction or --prompts_path must be specified.")
    
    prompts = []
    if args.prompts_path:
        prompts_file = Path(args.prompts_path)
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found at {args.prompts_path}")
        prompts = prompts_file.read_text(encoding="utf-8").splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        if not prompts:
            raise ValueError(f"Prompts file {args.prompts_path} is empty or contains only whitespace.")
    else:
        prompts = [args.instruction]
        if not prompts[0]:
            raise ValueError("Instruction prompt is empty.")
    
    main(args, prompts)