import dotenv

dotenv.load_dotenv(override=True)
from safetensors.torch import load_file, save_file

import os
from typing import List, Tuple
from safetensors.torch import load_file

from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage

import gradio as gr
import sys
import random
from transformers import Qwen2_5_VLModel as TextEncoder
from transformers import AutoTokenizer
import json
import argparse

global accelerator, pipeline, aspect_ratio


"""
## PIPELINE - 1

CUDA_VISIBLE_DEVICES=0 python3 -u gradio_qwen_image_aa_and_ab.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_1/snapshots/e3f588e48a4c67dff8dd173f5fb0343e86ddd405/aa_and_ab_qwen_image_1_LoRA_000007200.safetensors \
--tokenizer_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_1/snapshots/e3f588e48a4c67dff8dd173f5fb0343e86ddd405/tokenizer_0_aa_and_ab_qwen_image_1__000007200 \
--embeddings_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_1/snapshots/e3f588e48a4c67dff8dd173f5fb0343e86ddd405/text_encoder_0_aa_and_ab_qwen_image_1__000007200 \
--token_abstraction_json_path tokens.json \
--instructions_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/gradio_app_instructions/aa_and_ab.md \
--port 8900

--------

## PIPELINE - 2

CUDA_VISIBLE_DEVICES=0 python3 gradio_qwen_image_aa_and_ab.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_3/snapshots/203d95a4cfcf0a0e6447460731541ec2777d67cb/aa_and_ab_qwen_image_3_LoRA.safetensors \
--tokenizer_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_3/snapshots/203d95a4cfcf0a0e6447460731541ec2777d67cb/tokenizer_0_aa_and_ab_qwen_image_3_ \
--embeddings_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_3/snapshots/203d95a4cfcf0a0e6447460731541ec2777d67cb/text_encoder_0_aa_and_ab_qwen_image_3_ \
--token_abstraction_json_path tokens_3.json \
--instructions_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/gradio_app_instructions/aa_and_ab.md \
--port 8900

--------

## RDJ Pipeline - 1

CUDA_VISIBLE_DEVICES=0 python3 gradio_qwen_image_aa_and_ab.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/robert_downey_jr_ai/rdj_ai_LoRA.safetensors \
--tokenizer_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/robert_downey_jr_ai/tokenizer_0_rdj_ai_ \
--embeddings_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/robert_downey_jr_ai/text_encoder_0_rdj_ai_ \
--token_abstraction_json_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/output/robert_downey_jr_ai/tokens.json \
--instructions_path /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit/gradio_app_instructions/rdj.md \
--port 7878 

"""

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen-Image image generation script.")
    parser.add_argument(
        "--model_path", # "Qwen/Qwen-Image"
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str,
        default=None,
        help="Path to updated tokenizer.",
    )
    parser.add_argument(
        "--embeddings_path", 
        type=str,
        default=None,
        help="Path to embeddings file.",
    )
    parser.add_argument(
        "--token_abstraction_json_path",
        type=str,
        # required=True,
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
        "--instructions_path",
        type=str,
        default=None,
        help="Path to instructions markdown file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8900,
        help="Server port",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark",
        help="Negative prompt for generation."
    )

    return parser.parse_args()



args = parse_args()
with open(args.instructions_path, 'r') as md_file:
    einstructions = md_file.read()


def convert_lora_weights_before_load(args: argparse.Namespace, state_dict, new_path):
    if args.token_abstraction_json_path:
        popped_key = state_dict.pop("emb_params")
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key.replace("diffusion_model.", "transformer.")
        new_sd[new_key] = value

    save_file(new_sd, f"{new_path}")
    return new_sd




def load_pipeline(
    args, accelerator: Accelerator, weight_dtype: torch.dtype
) -> QwenImagePipeline:
    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)


    if args.transformer_lora_path:
        dir_path = '/'.join(args.transformer_lora_path.split('/')[:-1]) 
        cleaned_safetensors_file = args.transformer_lora_path.split('/')[-1].replace(".safetensors", "_cleaned.safetensors")
        new_path = f"{dir_path}/{cleaned_safetensors_file}"
        print(f"{new_path} exists")

        if not os.path.exists(new_path):
            print(f"Creating {new_path}...")
            state_dict = load_file(f"{args.transformer_lora_path}")
            state_dict = convert_lora_weights_before_load(args, state_dict, new_path)

        pipe.load_lora_weights(dir_path, weight_name=cleaned_safetensors_file)


    # loading new tokenizer and embeddings here!!!!
    if args.tokenizer_path and args.embeddings_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

        text_encoder = TextEncoder.from_pretrained(args.model_path,
                                                   subfolder="text_encoder",
                                                   torch_dtype=torch_dtype).to(device)

        main_state_dict = text_encoder.language_model.state_dict()

        state_dict = load_file(args.embeddings_path)["emb_params"].to(device)
        offset = len(tokenizer.get_vocab()) - state_dict.size(0)
        main_state_dict["embed_tokens.weight"] = torch.cat([main_state_dict["embed_tokens.weight"][:offset], state_dict])

        text_encoder.resize_token_embeddings(len(tokenizer))

        text_encoder.language_model.load_state_dict(main_state_dict)

        pipe.tokenizer = tokenizer

        pipe.text_encoder = text_encoder # verify the placement thoroughly to check if new tokens embeddings are loaded

    return pipe


def run(
    accelerator: Accelerator,
    pipeline: QwenImagePipeline,
    instruction: str,
    num_steps: int,
    width: int,
    height: int,
    text_guidance_scale: float,
    seed: int,
    num_images_per_prompt: int
) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    if seed == 0:
        seed = random.randint(1,sys.maxsize)
        print("Using random seed {seed}")
    print("Start Generation")

    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition." # for english prompt
    }

    # print(f"\n\n{width}, {height}\n\n")

    results = pipeline(
        num_images_per_prompt=num_images_per_prompt,
        prompt=instruction + positive_magic["en"],
        negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        width=width,
        height=height,
        num_inference_steps=num_steps,
        true_cfg_scale=text_guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(seed)
    )
    print("End Generation")
    return results


def init():
    global accelerator, pipeline
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16")

    # Set weight dtype
    weight_dtype = torch.bfloat16
    args = parse_args()
    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    pass


def run_generation(
    prompt, num_steps, width, height, text_gs, seed, numimgpp
):
    global accelerator, pipeline

    args = parse_args()

    if args.token_abstraction_json_path:
        with open(args.token_abstraction_json_path, "r") as file:
            representation_tokens = json.load(file)

        special_tokens = list(representation_tokens.keys())

        for special_token in special_tokens:
            prompt = prompt.replace(special_token, representation_tokens[special_token][0].replace(" ", ''))

    print(prompt)

    results = run(
        accelerator,
        pipeline,
        prompt,
        num_steps,
        width,
        height,
        text_gs,
        seed,
        numimgpp
    )
    output_images = results.images

    del results
    torch.cuda.empty_cache()
    
    return gr.update(visible=True, value=output_images)


def reset_all():
    return [
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None, visible=False),
        gr.update(value=""),
    ]


def show_alert():
    gr.Info("Running Model Inference, Please wait for output")


def set_aspect_ratio(name):
    global aspect_ratio 
    aspect_ratio = name
    return aspect_ratio

aspect_ratios_dict = {
    "1:1": (1024, 1024),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

with gr.Blocks() as demo:
    gr.Markdown("## Qwen-Image Image Generation with Custom TI LoRAs (Instructions and Tips at the bottom)")
    # with gr.Row():
    #     img1 = gr.Image(label="Input Image 1", interactive=True, type="pil")
    #     img2 = gr.Image(label="Input Image 2", interactive=True, type="pil")
    #     img3 = gr.Image(label="Input Image 3", interactive=True, type="pil")
    #     img4 = gr.Image(label="Input Image 4", interactive=True, type="pil")
    with gr.Row():
        prompt = gr.TextArea(label="Generation Prompt/Instruction")

    # with gr.Row():
    #     transformer_lora_path = gr.Textbox(label="Path to LoRA checkpoint directory")

    with gr.Row():
        num_steps = gr.Slider(
            1, 70, value=50, step=1, label="Number of inference steps"
        )

    with gr.Row():
        width = gr.Slider(480, 4096, value=1664, step=1, label="Output Image width")
        height = gr.Slider(480, 4096, value=928, step=1, label="Output Image height")

    with gr.Row():
        text_gs = gr.Slider(0.0, 10.0, value=4.0, step=0.1, label="Text Guidance Scale")
        # img_gs = gr.Slider(0.0, 10.0, value=2.0, step=0.1, label="Image Guidance Scale")

    with gr.Row():
        seed = gr.Slider(0, sys.maxsize, value=0, step=1, label="Random Seed")
        numimgpp = gr.Slider(1, 5, value=1, step=1, label="Number of images per prompt")

    # with gr.Row():
    #     dropdown = gr.Dropdown(
    #         choices=["16:9", "1:1", "9:16", "4:3", "3:4", "3:2", "2:3"],
    #         label="Aspect Ratio",
    #         value="16:9"  # default value
    #     )

    #     output = gr.Textbox(label="aspect ratio")

    #     dropdown.change(fn=set_aspect_ratio, inputs=dropdown, outputs=output)


    #     width = aspect_ratios_dict["16:9"][0]
    #     height = aspect_ratios_dict["16:9"][1]

    with gr.Row():
        reset_btn = gr.Button("Reset")
        edit_btn = gr.Button("Generate")

    with gr.Row():
        imgout = gr.Gallery(label="Generated Images", visible=False)

    with gr.Row():
        gr.Markdown(einstructions)

    reset_btn.click(
        # reset_all, inputs=[], outputs=[img1, img2, img3, img4, imgout, prompt]
        reset_all, inputs=[], outputs=[imgout, prompt]
    )
    
    edit_btn.click(show_alert, inputs=None, outputs=None).success(
        run_generation,
        inputs=[
            prompt,
            num_steps,
            width,
            height,
            text_gs,
            seed,
            numimgpp
        ],
        outputs=[imgout],
    )

init()
args = parse_args()
demo.launch(server_name="0.0.0.0", server_port=args.port) 
