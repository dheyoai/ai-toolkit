# from diffusers import DiffusionPipeline
# import torch
# from safetensors.torch import load_file, save_file
# from transformers import AutoTokenizer 
# # Fixed import - use correct text encoder class
# from transformers import Qwen2VLForConditionalGeneration as TextEncoder
# import json
# from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
# import argparse
# from pathlib import Path
# import os
# from typing import List
# import time 

# '''
# HIP_VISIBLE_DEVICES=3 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
# --transformer_lora_path /path/to/your/lora.safetensors \
# --num_inference_steps 50 \
# --output_image_path inferenced_images/output.png \
# --prompts_path prompts.txt \
# --aspect_ratio "16:9" \
# --num_images_per_prompt 8
# '''


# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Qwen-Image image generation script.")
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         required=True,
#         help="Path to model checkpoint.",
#     )
#     parser.add_argument(
#         "--tokenizer_path",
#         type=str,
#         default=None,
#         help="Path to updated tokenizer (optional - uses base model if not provided).",
#     )
#     parser.add_argument(
#         "--text_encoder_path",
#         type=str,
#         default=None,
#         help="Path to text encoder checkpoint (optional - uses base model if not provided).",
#     )
#     parser.add_argument(
#         "--token_abstraction_json_path",
#         type=str,
#         default=None,
#         help="Path to token abstraction dict",
#     )
#     parser.add_argument(
#         "--transformer_lora_path",
#         type=str,
#         default=None,
#         help="Path to transformer LoRA checkpoint.",
#     )
#     parser.add_argument(
#         "--num_inference_steps",
#         type=int,
#         default=50,
#         help="Number of inference steps."
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for generation."
#     )
#     parser.add_argument(
#         "--dtype",
#         type=str,
#         default='bf16',
#         choices=['fp32', 'fp16', 'bf16'],
#         help="Data type for model weights."
#     )
#     parser.add_argument(
#         "--true_cfg_scale",
#         type=float,
#         default=4.0,
#         help="Text guidance scale."
#     )
#     parser.add_argument(
#         "--instruction",
#         type=str,
#         default=None,
#         help="Text prompt for generation."
#     )
#     parser.add_argument(
#         "--negative_prompt",
#         type=str,
#         default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark",
#         help="Negative prompt for generation."
#     )
#     parser.add_argument(
#         "--output_image_path",
#         type=str,
#         default="output.png",
#         help="Path to save output image."
#     )
#     parser.add_argument(
#         "--num_images_per_prompt",
#         type=int,
#         default=1,
#         help="Number of images to generate per prompt."
#     )
#     parser.add_argument(
#         "--prompts_path",
#         type=str,
#         default=None,
#         help="Path to prompts.txt for bulk generation",
#     )
#     parser.add_argument(
#         "--aspect_ratio",
#         type=str,
#         default="16:9",
#         help="Aspect ratio for generated images (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)",
#     )

#     return parser.parse_args()


# def convert_lora_weights_before_load(state_dict, new_path):
#     """Convert LoRA weights to proper format and save cleaned version."""
#     # Remove embedding parameters if they exist
#     if "emb_params" in state_dict:
#         popped_key = state_dict.pop("emb_params")
#         print(f"Removed emb_params from state_dict")
    
#     # Convert keys from diffusion_model to transformer format
#     new_sd = {}
#     for key, value in state_dict.items():
#         if key.startswith("diffusion_model."):
#             new_key = key.replace("diffusion_model.", "transformer.")
#         else:
#             new_key = key
#         new_sd[new_key] = value
    
#     # Save cleaned weights
#     save_file(new_sd, new_path)
#     print(f"Saved cleaned LoRA weights to: {new_path}")
#     return new_sd


# def load_pipeline(args: argparse.Namespace):
#     """Load and configure the Qwen-Image pipeline with LoRA and custom components."""
#     # Set device and dtype
#     if torch.cuda.is_available():
#         torch_dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)
#         device = "cuda"
#     else:
#         torch_dtype = torch.float32
#         device = "cpu"
    
#     print(f"Using device: {device}, dtype: {torch_dtype}")

#     # Load base pipeline
#     try:
#         pipe = QwenImagePipeline.from_pretrained(
#             args.model_path, 
#             torch_dtype=torch_dtype,
#             use_safetensors=True
#         )
#         pipe = pipe.to(device)
#         print(f"Successfully loaded base pipeline from {args.model_path}")
#     except Exception as e:
#         print(f"Error loading base pipeline: {e}")
#         raise

#     # Load LoRA weights if provided
#     if args.transformer_lora_path:
#         try:
#             dir_path = os.path.dirname(args.transformer_lora_path)
#             original_filename = os.path.basename(args.transformer_lora_path)
#             cleaned_filename = original_filename.replace(".safetensors", "_cleaned.safetensors")
#             cleaned_path = os.path.join(dir_path, cleaned_filename)
            
#             print(f"LoRA path: {args.transformer_lora_path}")
#             print(f"Cleaned path: {cleaned_path}")

#             # Convert LoRA weights if cleaned version doesn't exist
#             if not os.path.exists(cleaned_path):
#                 print(f"Creating cleaned LoRA weights...")
#                 state_dict = load_file(args.transformer_lora_path)
#                 convert_lora_weights_before_load(state_dict, cleaned_path)
#             else:
#                 print(f"Using existing cleaned LoRA weights: {cleaned_path}")

#             # Load LoRA weights
#             pipe.load_lora_weights(dir_path, weight_name=cleaned_filename)
#             print(f"Successfully loaded LoRA weights")
            
#         except Exception as e:
#             print(f"Error loading LoRA weights: {e}")
#             print("Continuing without LoRA...")

#     # Load custom tokenizer if path provided
#     if args.tokenizer_path:
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(
#                 args.tokenizer_path,
#                 trust_remote_code=True
#             )
#             pipe.tokenizer = tokenizer
#             print(f"Successfully loaded custom tokenizer from {args.tokenizer_path}")
#             print(f"Tokenizer vocab size: {len(tokenizer)}")
#         except Exception as e:
#             print(f"Error loading custom tokenizer: {e}")
#             print("Using base model tokenizer...")
#     else:
#         print("No custom tokenizer path provided, using base model tokenizer")

#     # Load custom text encoder if path provided
#     if args.text_encoder_path:
#         try:
#             text_encoder = TextEncoder.from_pretrained(
#                 args.text_encoder_path,
#                 ignore_mismatched_sizes=True,
#                 torch_dtype=torch_dtype,
#                 trust_remote_code=True
#             ).to(device)
            
#             # Resize token embeddings to match tokenizer
#             if hasattr(text_encoder, 'resize_token_embeddings'):
#                 text_encoder.resize_token_embeddings(len(pipe.tokenizer))
#                 print(f"Resized text encoder embeddings to {len(pipe.tokenizer)}")
            
#             pipe.text_encoder = text_encoder
#             print(f"Successfully loaded custom text encoder from {args.text_encoder_path}")
            
#         except Exception as e:
#             print(f"Error loading custom text encoder: {e}")
#             print("Using base model text encoder...")
#     else:
#         print("No custom text encoder path provided, using base model text encoder")

#     return pipe


# def main(args: argparse.Namespace, prompts: List[str]) -> None:
#     """Main inference function."""
#     # Load pipeline with all components
#     pipe = load_pipeline(args)
    
#     # Magic prompts for quality enhancement
#     positive_magic = {
#         "en": "Ultra HD, 4K, cinematic composition."
#     }

#     # Load token abstractions if provided
#     representation_tokens = {}
#     if args.token_abstraction_json_path and os.path.exists(args.token_abstraction_json_path):
#         try:
#             with open(args.token_abstraction_json_path, "r") as file:
#                 representation_tokens = json.load(file)
#             print(f"Loaded token abstractions: {list(representation_tokens.keys())}")
#         except Exception as e:
#             print(f"Error loading token abstractions: {e}")

#     special_tokens = list(representation_tokens.keys())

#     # Aspect ratio configurations
#     aspect_ratios = {
#         "1:1": (1024, 1024),
#         "16:9": (1664, 928),
#         "9:16": (928, 1664),
#         "4:3": (1472, 1140),
#         "3:4": (1140, 1472),
#         "3:2": (1584, 1056),
#         "2:3": (1056, 1584),
#     }

#     if args.aspect_ratio not in aspect_ratios:
#         print(f"Warning: Invalid aspect ratio {args.aspect_ratio}. Using 16:9")
#         args.aspect_ratio = "16:9"
    
#     width, height = aspect_ratios[args.aspect_ratio]
#     print(f"Using aspect ratio {args.aspect_ratio}: {width}x{height}")

#     # Generate images for each prompt
#     for idx, prompt in enumerate(prompts):
#         # Replace special tokens
#         original_prompt = prompt
#         for special_token in special_tokens:
#             if special_token in prompt:
#                 replacement = representation_tokens[special_token][0].replace(" ", "")
#                 prompt = prompt.replace(special_token, replacement)
#                 print(f"Replaced '{special_token}' with '{replacement}'")

#         final_prompt = prompt + " " + positive_magic["en"]
#         print(f"Prompt {idx + 1}/{len(prompts)}: {final_prompt}")

#         try:
#             # Generate images
#             generator = torch.Generator(device=pipe.device if hasattr(pipe, 'device') else "cuda")
#             generator.manual_seed(args.seed + idx)  # Different seed for each prompt
            
#             images = pipe(
#                 num_images_per_prompt=args.num_images_per_prompt,
#                 prompt=final_prompt,
#                 negative_prompt=args.negative_prompt,
#                 width=width,
#                 height=height,
#                 num_inference_steps=args.num_inference_steps,
#                 true_cfg_scale=args.true_cfg_scale,
#                 generator=generator
#             ).images

#             # Save images
#             os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
#             timestamp = time.strftime("%d-%m-%y_%H-%M-%S")

#             for image_id, image in enumerate(images):
#                 base_name = args.output_image_path.replace('.png', '')
#                 file_path = f"{base_name}_prompt{idx:03d}_img{image_id:02d}_{timestamp}.png"
#                 image.save(file_path)
#                 print(f"✓ Saved: {file_path}")
                
#         except Exception as e:
#             print(f"✗ Error generating images for prompt {idx + 1}: {e}")
#             continue

#     print(f"Generation complete! Processed {len(prompts)} prompts.")


# if __name__ == '__main__':
#     args = parse_args()
    
#     # Validate required arguments
#     if not args.instruction and not args.prompts_path:
#         raise ValueError("Either --instruction or --prompts_path must be specified")

#     # Load prompts
#     if args.prompts_path:
#         if not os.path.exists(args.prompts_path):
#             raise FileNotFoundError(f"Prompts file not found: {args.prompts_path}")
#         prompts_path = Path(args.prompts_path)
#         prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
#         print(f"Loaded {len(prompts)} prompts from {args.prompts_path}")
#     else:
#         prompts = [args.instruction]
#         print(f"Using single instruction prompt")

#     if not prompts:
#         raise ValueError("No valid prompts found")
    
#     main(args, prompts)

from diffusers import DiffusionPipeline
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer 
# Fixed import - use correct text encoder class
from transformers import Qwen2VLForConditionalGeneration as TextEncoder
import json
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
import argparse
from pathlib import Path
import os
from typing import List
import time 

'''
HIP_VISIBLE_DEVICES=3 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path /path/to/your/lora.safetensors \
--num_inference_steps 50 \
--output_image_path inferenced_images/output_directory \
--prompts_path prompts.txt \
--aspect_ratio "16:9" \
--num_images_per_prompt 8
'''


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen-Image image generation script.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to updated tokenizer (optional - uses base model if not provided).",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to text encoder checkpoint (optional - uses base model if not provided).",
    )
    parser.add_argument(
        "--token_abstraction_json_path",
        type=str,
        default=None,
        help="Path to token abstraction dict",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA checkpoint.",
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
        default="output",
        help="Directory path to save output images (will be created if it doesn't exist)."
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
        help="Aspect ratio for generated images (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)",
    )

    return parser.parse_args()


def convert_lora_weights_before_load(state_dict, new_path):
    """Convert LoRA weights to proper format and save cleaned version."""
    # Remove embedding parameters if they exist
    if "emb_params" in state_dict:
        popped_key = state_dict.pop("emb_params")
        print(f"Removed emb_params from state_dict")
    
    # Convert keys from diffusion_model to transformer format
    new_sd = {}
    for key, value in state_dict.items():
        if key.startswith("diffusion_model."):
            new_key = key.replace("diffusion_model.", "transformer.")
        else:
            new_key = key
        new_sd[new_key] = value
    
    # Save cleaned weights
    save_file(new_sd, new_path)
    print(f"Saved cleaned LoRA weights to: {new_path}")
    return new_sd


def load_pipeline(args: argparse.Namespace):
    """Load and configure the Qwen-Image pipeline with LoRA and custom components."""
    # Set device and dtype
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    print(f"Using device: {device}, dtype: {torch_dtype}")

    # Load base pipeline
    try:
        pipe = QwenImagePipeline.from_pretrained(
            args.model_path, 
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        print(f"Successfully loaded base pipeline from {args.model_path}")
    except Exception as e:
        print(f"Error loading base pipeline: {e}")
        raise

    # Load LoRA weights if provided
    if args.transformer_lora_path:
        try:
            dir_path = os.path.dirname(args.transformer_lora_path)
            original_filename = os.path.basename(args.transformer_lora_path)
            cleaned_filename = original_filename.replace(".safetensors", "_cleaned.safetensors")
            cleaned_path = os.path.join(dir_path, cleaned_filename)
            
            print(f"LoRA path: {args.transformer_lora_path}")
            print(f"Cleaned path: {cleaned_path}")

            # Convert LoRA weights if cleaned version doesn't exist
            if not os.path.exists(cleaned_path):
                print(f"Creating cleaned LoRA weights...")
                state_dict = load_file(args.transformer_lora_path)
                convert_lora_weights_before_load(state_dict, cleaned_path)
            else:
                print(f"Using existing cleaned LoRA weights: {cleaned_path}")

            # Load LoRA weights
            pipe.load_lora_weights(dir_path, weight_name=cleaned_filename)
            print(f"Successfully loaded LoRA weights")
            
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            print("Continuing without LoRA...")

    # Load custom tokenizer if path provided
    if args.tokenizer_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                trust_remote_code=True
            )
            pipe.tokenizer = tokenizer
            print(f"Successfully loaded custom tokenizer from {args.tokenizer_path}")
            print(f"Tokenizer vocab size: {len(tokenizer)}")
        except Exception as e:
            print(f"Error loading custom tokenizer: {e}")
            print("Using base model tokenizer...")
    else:
        print("No custom tokenizer path provided, using base model tokenizer")

    # Load custom text encoder if path provided
    if args.text_encoder_path:
        try:
            text_encoder = TextEncoder.from_pretrained(
                args.text_encoder_path,
                ignore_mismatched_sizes=True,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(device)
            
            # Resize token embeddings to match tokenizer
            if hasattr(text_encoder, 'resize_token_embeddings'):
                text_encoder.resize_token_embeddings(len(pipe.tokenizer))
                print(f"Resized text encoder embeddings to {len(pipe.tokenizer)}")
            
            pipe.text_encoder = text_encoder
            print(f"Successfully loaded custom text encoder from {args.text_encoder_path}")
            
        except Exception as e:
            print(f"Error loading custom text encoder: {e}")
            print("Using base model text encoder...")
    else:
        print("No custom text encoder path provided, using base model text encoder")

    return pipe


def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """Sanitize filename by removing invalid characters and limiting length."""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return filename


def main(args: argparse.Namespace, prompts: List[str]) -> None:
    """Main inference function."""
    # Load pipeline with all components
    pipe = load_pipeline(args)
    
    # Magic prompts for quality enhancement
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition."
    }

    # Load token abstractions if provided
    representation_tokens = {}
    if args.token_abstraction_json_path and os.path.exists(args.token_abstraction_json_path):
        try:
            with open(args.token_abstraction_json_path, "r") as file:
                representation_tokens = json.load(file)
            print(f"Loaded token abstractions: {list(representation_tokens.keys())}")
        except Exception as e:
            print(f"Error loading token abstractions: {e}")

    special_tokens = list(representation_tokens.keys())

    # Aspect ratio configurations
    aspect_ratios = {
        "1:1": (1024, 1024),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    if args.aspect_ratio not in aspect_ratios:
        print(f"Warning: Invalid aspect ratio {args.aspect_ratio}. Using 16:9")
        args.aspect_ratio = "16:9"
    
    width, height = aspect_ratios[args.aspect_ratio]
    print(f"Using aspect ratio {args.aspect_ratio}: {width}x{height}")

    # Create output directory - treat output_image_path as directory
    output_dir = args.output_image_path
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate a timestamp for this run
    run_timestamp = time.strftime("%d-%m-%y_%H-%M-%S")
    
    total_images_generated = 0

    # Generate images for each prompt
    for idx, prompt in enumerate(prompts):
        # Replace special tokens
        original_prompt = prompt
        for special_token in special_tokens:
            if special_token in prompt:
                replacement = representation_tokens[special_token][0].replace(" ", "")
                prompt = prompt.replace(special_token, replacement)
                print(f"Replaced '{special_token}' with '{replacement}'")

        final_prompt = prompt + " " + positive_magic["en"]
        print(f"\nPrompt {idx + 1}/{len(prompts)}: {final_prompt}")

        # Create a sanitized prompt name for filenames
        prompt_name = sanitize_filename(original_prompt.replace(" ", "_"))

        try:
            # Generate images
            generator = torch.Generator(device=pipe.device if hasattr(pipe, 'device') else "cuda")
            generator.manual_seed(args.seed + idx)  # Different seed for each prompt
            
            images = pipe(
                num_images_per_prompt=args.num_images_per_prompt,
                prompt=final_prompt,
                negative_prompt=args.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=generator
            ).images

            # Save images for this prompt
            for image_id, image in enumerate(images):
                filename = f"prompt{idx+1:03d}_{prompt_name}_img{image_id+1:02d}_{run_timestamp}.png"
                file_path = os.path.join(output_dir, filename)
                image.save(file_path)
                total_images_generated += 1
                print(f"✓ Saved: {filename}")
                
        except Exception as e:
            print(f"✗ Error generating images for prompt {idx + 1}: {e}")
            continue

    print(f"\n=== Generation Complete ===")
    print(f"Processed: {len(prompts)} prompts")
    print(f"Generated: {total_images_generated} images")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    
    # Validate required arguments
    if not args.instruction and not args.prompts_path:
        raise ValueError("Either --instruction or --prompts_path must be specified")

    # Load prompts
    if args.prompts_path:
        if not os.path.exists(args.prompts_path):
            raise FileNotFoundError(f"Prompts file not found: {args.prompts_path}")
        prompts_path = Path(args.prompts_path)
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_path}")
    else:
        prompts = [args.instruction]
        print(f"Using single instruction prompt")

    if not prompts:
        raise ValueError("No valid prompts found")
    
    main(args, prompts)