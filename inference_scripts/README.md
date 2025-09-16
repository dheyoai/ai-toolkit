# Qwen-Image LoRA Inference

This script generates images using the Qwen-Image model from Hugging Face Diffusers, with support for LoRA weights, custom tokenizer, and text encoder. It leverages Intel Extension for PyTorch (IPEX) for CPU optimizations on Intel Xeon systems.

## Prerequisites
- **System**: Ubuntu 24.04 LTS
- **Hardware**: Intel Xeon CPU (e.g., Xeon 6975P-C with 96 cores, 1.5TiB RAM)
- **Python**: 3.12
- **Dependencies**: Specified in `requirements.txt`

## Installation

1. **Set Up Python Environment**
   Install Python 3.12 and create a virtual environment:
   ```bash
   sudo apt-get update
   sudo apt-get install python3.12 python3.12-venv
   python3.12 -m venv .env
   source .env/bin/activate
   ```

2. **Install Dependencies**
   Install all required packages using `requirements.txt`:
   ```bash
   pip install -r ipex_requirements.txt
   ```

## Usage

1. **Run the Script**
   ```bash
   python3 infer_qwen_image_with_lora.py \
       --model_path "Qwen/Qwen-Image" \
       --transformer_lora_path /path/to/lora.safetensors \
       --tokenizer_path /path/to/tokenizer/dir \
       --text_encoder_path /path/to/text_encoder/dir \
       --token_abstraction_json_path /path/to/tokens.json \
       --num_inference_steps 50 \
       --output_image_path /path/to/output/image_scene.png \
       --prompts_path /path/to/prompt.txt \
       --aspect_ratio "16:9" \
       --num_images_per_prompt 1
   ```
   - **Output**: Images are saved in `/path/to/output/` with names like `image_scene_0_0_DD-MM-YY_HH-MM-SS.png`.

## Configuration Options
- `--model_path`: Qwen-Image model (e.g., "Qwen/Qwen-Image" or local path).
- `--transformer_lora_path`: LoRA weights for the transformer (.safetensors).
- `--tokenizer_path`, `--text_encoder_path`: Custom tokenizer and text encoder checkpoints.
- `--token_abstraction_json_path`: JSON file mapping special tokens to replacements.
- `--num_inference_steps`: Denoising steps (default: 50).
- `--output_image_path`: Base path for output images.
- `--prompts_path`: Text file with prompts (one per line).
- `--aspect_ratio`: Image aspect ratio (e.g., "16:9" for 1664x928).
- `--num_images_per_prompt`: Images per prompt (default: 1).


## Notes
- The script uses IPEX for CPU optimizations, leveraging the Xeon 6975P-C’s AVX512, VNNI, and AMX features.
- Expected runtime is ~36 minutes per image at 1664x928 with 50 steps. Contact `aakash.varma@dheyo.ai` for performance optimization guidance if needed.