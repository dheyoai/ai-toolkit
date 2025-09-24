from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file
import torch
import argparse
import os

def main(args):
    # Load the pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load the converted LoRA weights
    pipe.load_lora_weights(args.transformer_lora_path)

    # Load the [AB] embedding
    embedding_path = "/dheyo/varunika/output/lora_ab_base/[AB]_000004319.safetensors"
    embedding_state_dict = load_file(embedding_path)
    pipe.tokenizer.add_tokens(["[AB]"])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder.get_input_embeddings().weight.data[-1] = embedding_state_dict["emb_params"][0]

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_image_path)
    os.makedirs(output_dir, exist_ok=True)

    # Generate image
    height, width = map(int, args.aspect_ratio.split(":"))
    height = 576  # Scaled 16:9 for efficiency
    width = 1024
    image = pipe(
        prompt=args.instruction,
        num_inference_steps=args.num_inference_steps,
        height=height,
        width=width,
        guidance_scale=7.5
    ).images[0]
    image.save(args.output_image_path)
    print(f"Image saved to {args.output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen-Image")
    parser.add_argument("--transformer_lora_path", required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output_image_path", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--aspect_ratio", default="16:9")
    args = parser.parse_args()
    main(args)