import argparse
import sys
from pathlib import Path
from typing import List

from core.job_orchestrator import JobOrchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Generation Job System")
    
    parser.add_argument("--job_name", type=str, required=True, help="Job name")
    parser.add_argument("--model_type", type=str, default="qwen", choices=["qwen"], help="Model type")
    parser.add_argument("--model_path", type=str, required=True, help="Model HF ID or local path")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hf_lora_id", type=str, help="HuggingFace LoRA repo ID")
    group.add_argument("--local_lora_config", action="store_true", help="Use local LoRA paths")
    
    parser.add_argument("--transformer_lora_path", type=str, help="Local LoRA weights path")
    parser.add_argument("--tokenizer_path", type=str, help="Local tokenizer path")
    parser.add_argument("--embeddings_path", type=str, help="Local embeddings path")
    parser.add_argument("--token_abstraction_json_path", type=str, help="Token mapping JSON")
    
    parser.add_argument("--instruction", type=str, help="Single prompt")
    parser.add_argument("--prompts", type=str, nargs="+", help="Multiple prompts")
    parser.add_argument("--prompts_path", type=str, help="File with prompts")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--negative_prompt", type=str, 
                       default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark")
    
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--cache_dir", type=str, default="/dheyo/lora-infer", help="Cache directory for models and LoRA")
    
    parser.add_argument("--cleanup_local", action="store_true", help="Cleanup local files after generation")
    parser.add_argument("--cleanup_cache", action="store_true", help="Cleanup model cache")
    
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_path:
        return Path(args.prompts_path).read_text().strip().splitlines()
    elif args.prompts:
        return args.prompts
    elif args.instruction:
        return [args.instruction]
    else:
        raise ValueError("Must provide --instruction, --prompts, or --prompts_path")


def validate_args(args: argparse.Namespace):
    if args.local_lora_config:
        required = ["transformer_lora_path", "tokenizer_path", "embeddings_path"]
        missing = [arg for arg in required if not getattr(args, arg)]
        if missing:
            raise ValueError(f"With --local_lora_config, must provide: {', '.join(missing)}")


def main():
    try:
        args = parse_args()
        validate_args(args)
        prompts = load_prompts(args)
        
        logger.info(f"Starting job '{args.job_name}' with {len(prompts)} prompts")
        
        orchestrator = JobOrchestrator(args)
        success = orchestrator.run_job(prompts)
        
        if success:
            logger.info("Job completed successfully")
            sys.exit(0)
        else:
            logger.error("Job failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
