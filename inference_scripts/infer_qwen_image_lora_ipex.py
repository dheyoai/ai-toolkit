import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import intel_extension_for_pytorch as ipex
import torch
from diffusers import QwenImagePipeline
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLModel as TextEncoder


def setup_optimization(optimization_mode: str):
    """Setup environment variables for optimization"""
    if optimization_mode == "ipex":
        env_vars = {
            "OMP_NUM_THREADS": "96",
            "MKL_NUM_THREADS": "96",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "KMP_BLOCKTIME": "1"
        }
    elif optimization_mode == "amx":
        env_vars = {
            "OMP_NUM_THREADS": "96",
            "MKL_NUM_THREADS": "96",
            "MKL_ENABLE_INSTRUCTIONS": "AMX_BF16,AMX_INT8,AVX512_E1",
            "MKL_CBWR": "AMX",
            "MKL_DYNAMIC": "FALSE",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "KMP_BLOCKTIME": "0"
        }
    else:  # none
        return
    
    for key, value in env_vars.items():
        os.environ[key] = str(value)
    
    torch.set_num_threads(int(env_vars["OMP_NUM_THREADS"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image image generation script with LoRA support.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen-Image model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the updated tokenizer checkpoint")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to the text encoder checkpoint")
    parser.add_argument("--token_abstraction_json_path", type=str, default=None, help="Path to the token abstraction dictionary JSON file")
    parser.add_argument("--transformer_lora_path", type=str, default=None, help="Path to the transformer LoRA checkpoint")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Data type for model weights")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale for text conditioning")
    parser.add_argument("--instruction", type=str, default=None, help="Text prompt for image generation (single prompt mode)")
    parser.add_argument("--negative_prompt", type=str, default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark", help="Negative prompt to avoid in generation")
    parser.add_argument("--output_image_path", type=str, default="output.png", help="Base path to save generated images")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt")
    parser.add_argument("--prompts_path", type=str, default=None, help="Path to a text file containing multiple prompts")
    parser.add_argument("--aspect_ratio", type=str, default="16:9", help="Aspect ratio for generated images")
    parser.add_argument("--optimization", type=str, default="ipex", choices=["ipex", "amx", "none"], 
                       help="Optimization mode: ipex (recommended), amx (explicit AMX), none (no optimization)")
    
    return parser.parse_args()


def convert_lora_weights_before_load(args: argparse.Namespace, state_dict: dict, new_path: str) -> dict:
    if args.token_abstraction_json_path:
        state_dict.pop("emb_params", None)

    new_sd = {}
    for key, value in state_dict.items():
        new_key = key.replace("diffusion_model.", "transformer.")
        new_sd[new_key] = value

    save_file(new_sd, new_path)
    return new_sd


def load_pipeline(args: argparse.Namespace) -> tuple[QwenImagePipeline, str]:
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        device = "cuda"
    else:
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        device = "cpu"

    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    if args.transformer_lora_path:
        dir_path = "/".join(args.transformer_lora_path.split("/")[:-1])
        original_file = args.transformer_lora_path.split("/")[-1]
        cleaned_safetensors_file = original_file.replace(".safetensors", "_cleaned.safetensors")
        new_path = f"{dir_path}/{cleaned_safetensors_file}"

        print(f"Checking for {new_path}...")
        if not os.path.exists(new_path):
            print(f"Creating {new_path}...")
            state_dict = load_file(args.transformer_lora_path)
            convert_lora_weights_before_load(args, state_dict, new_path)

        pipe.load_lora_weights(dir_path, weight_name=cleaned_safetensors_file)

    if args.tokenizer_path and args.text_encoder_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        text_encoder = TextEncoder.from_pretrained(
            args.text_encoder_path,
            ignore_mismatched_sizes=True,
            torch_dtype=torch_dtype,
        ).to(device)

        text_encoder.resize_token_embeddings(len(tokenizer))
        pipe.tokenizer = tokenizer
        pipe.text_encoder = text_encoder

    # Apply IPEX optimization for both ipex and amx modes
    if args.optimization in ["ipex", "amx"] and args.dtype == "bf16":
        pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
        pipe.transformer = ipex.optimize(pipe.transformer.eval(), dtype=torch.bfloat16, inplace=True)
        pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)

    return pipe, device


def main(args: argparse.Namespace, prompts: List[str]) -> None:
    setup_optimization(args.optimization)
    
    pipe, device = load_pipeline(args)

    positive_magic = {"en": "Ultra HD, 4K, cinematic composition."}

    representation_tokens = {}
    if args.token_abstraction_json_path:
        with open(args.token_abstraction_json_path, "r") as file:
            representation_tokens = json.load(file)
        special_tokens = list(representation_tokens.keys())

    aspect_ratios = {
        "1:1": (1024, 1024),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratios.get(args.aspect_ratio, (1664, 928))

    for idx, prompt in enumerate(prompts):
        if representation_tokens:
            for special_token in special_tokens:
                prompt = prompt.replace(special_token, representation_tokens[special_token][0].replace(" ", ""))

        print(f"Generating for prompt: {prompt}")
        start_time = time.time()

        if args.dtype == "bf16":
            with torch.amp.autocast("cpu", enabled=True, dtype=torch.bfloat16):
                images = pipe(
                    num_images_per_prompt=args.num_images_per_prompt,
                    prompt=prompt + positive_magic["en"],
                    negative_prompt=args.negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.true_cfg_scale,
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                ).images
        else:
            images = pipe(
                num_images_per_prompt=args.num_images_per_prompt,
                prompt=prompt + positive_magic["en"],
                negative_prompt=args.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            ).images

        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.1f}s ({generation_time/60:.1f}m)")

        output_dir = os.path.dirname(args.output_image_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%d-%m-%y_%H-%M-%S")

        for image_id, image in enumerate(images):
            file_path = f"{args.output_image_path.replace('.png', '')}_{idx}_{image_id}_{timestamp}.png"
            image.save(file_path)
            print(f"Saved {file_path}")


if __name__ == "__main__":
    args = parse_args()
    if not args.instruction and not args.prompts_path:
        raise ValueError("Either --instruction or --prompts_path must be specified.")

    if args.prompts_path:
        prompts_path = Path(args.prompts_path)
        prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    else:
        prompts = [args.instruction]

    main(args, prompts)