# AI Toolkit - Inference Toolkit

## Quick Start

### Usage

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py \
  --job_name "aa_ab_base_test" \
  --model_type qwen \
  --model_path "Qwen/Qwen-Image" \
  --local_lora_config \
  --transformer_lora_path "/workspace/aakashvarma/ai-toolkit/output/ab_rc_base/ab_rc_base_LoRA_000005604.safetensors" \
  --tokenizer_path "/workspace/aakashvarma/ai-toolkit/output/ab_rc_base/tokenizer_0_ab_rc_base__000005604" \
  --embeddings_path "/workspace/aakashvarma/ai-toolkit/output/ab_rc_base/[AB][RC]_000005604.safetensors" \
  --token_abstraction_json_path "/workspace/aakashvarma/ai-toolkit/output/ab_rc_base/tokens.json" \
  --instruction "A close-up portrait of (([RC] man)) and (([AB] woman)) sitting closely together in a cozy cafe, warm ambient lighting, soft bokeh background, both facing the camera with gentle smiles, intimate and natural expression" \
  --num_inference_steps 50 \
  --aspect_ratio "16:9" \
  --num_images_per_prompt 1
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-toolkit/inference_toolkit
   ```

2. **Install dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python3 main.py --help
   ```

## Command Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--job_name` | Unique name for the generation job | `"my_job"` |
| `--model_path` | HuggingFace model ID or local path | `"Qwen/Qwen-Image"` |

### Model Configuration

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model_type` | Type of model to use | `qwen` | `qwen` |
| `--dtype` | Model precision | `bf16` | `fp32`, `fp16`, `bf16` |

### LoRA Configuration

Choose one of the following LoRA options:

#### Option 1: HuggingFace LoRA
```bash
--hf_lora_id "username/lora-repo"
```

#### Option 2: Local LoRA (requires all paths)
```bash
--local_lora_config \
--transformer_lora_path "/path/to/lora.safetensors" \
--tokenizer_path "/path/to/tokenizer" \
--embeddings_path "/path/to/embeddings.safetensors" \
--token_abstraction_json_path "/path/to/tokens.json"
```

### Prompt Configuration

Choose one of the following prompt options:

| Argument | Description | Example |
|----------|-------------|---------|
| `--instruction` | Single prompt | `"A cat sitting on a chair"` |
| `--prompts` | Multiple prompts | `"prompt1" "prompt2" "prompt3"` |
| `--prompts_path` | File with prompts (one per line) | `"/path/to/prompts.txt"` |

### Generation Parameters

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--num_images_per_prompt` | Images per prompt | `1` | Any integer |
| `--num_inference_steps` | Denoising steps | `50` | Any integer |
| `--true_cfg_scale` | Guidance scale | `4.0` | Any float |
| `--aspect_ratio` | Image aspect ratio | `16:9` | `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3` |
| `--seed` | Random seed | `42` | Any integer |
| `--negative_prompt` | Negative prompt | Default negative prompt | Any string |

### Output Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--output_dir` | Output directory | `./outputs` |

### Cleanup Options

| Argument | Description |
|----------|-------------|
| `--cleanup_local` | Delete local files after generation |
| `--cleanup_cache` | Delete model cache after job completion |

## Redis Job Processing

The toolkit supports Redis-based job processing for distributed workflows with S3 upload.

### Prerequisites

1. **Set AWS environment variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-west-1
   export S3_BUCKET_NAME=dheyo-generations
   ```

### Running Redis Tests

#### Step 1: Submit Job Request
```bash
cd tests
python3 producer.py
```

#### Step 2: Start Job Process
```bash
cd tests
HIP_VISIBLE_DEVICES=2 python3 consumer.py
```

### S3 Upload Structure

Generated images are uploaded to S3 with this structure:
```
s3://dheyo-generations/
└── images/
    └── user_{job_name}/
        └── {generation_id}/
            ├── output_{uuid1}.png
            └── output_{uuid2}.png
```

## Integration with Dwarpaal

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-toolkit/inference_toolkit
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv aitool
   source aitool/bin/activate
   ```

3. **Install dependencies**:
   
   **AMD:**
   ```bash
   uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
   pip install -r ../requirements.txt
   ```
   
   **NVIDIA:**
   ```bash
   pip install -r ../requirements.txt
   ```

4. **Set Environment Variables**:
   
   Edit the file called `setup_env.sh` with the following content:
   ```bash
   # Redis Configuration
   export REDIS_HOST=your-redis-host
   export REDIS_PORT=6379
   export REDIS_PASSWORD='your-redis-password'
   export REDIS_DB=0

   # Redis Queue Names
   export REDIS_REQUEST_QUEUE=lora_generations
   export REDIS_RESPONSE_QUEUE=generations_status
   export REDIS_TIMEOUT="5"

   # AWS S3 Configuration
   export UPLOADER_TYPE="s3"
   export AWS_ACCESS_KEY_ID=your-access-key-id
   export AWS_SECRET_ACCESS_KEY=your-secret-access-key
   export AWS_DEFAULT_REGION=us-west-1
   export S3_BUCKET_NAME=dheyo-generations
   export S3_KEY_PREFIX="images"

   # Default Directories
   export DEFAULT_OUTPUT_DIR="/dheyo/lora-infer/outputs"
   export DEFAULT_CACHE_DIR="/dheyo/lora-infer/cache"
   ```
   
   Then source the file:
   ```bash
   source setup_env.sh
   ```

5. **Run Consumer**:
   ```bash
   cd tests
   HIP_VISIBLE_DEVICES=2 python3 consumer.py
   ```