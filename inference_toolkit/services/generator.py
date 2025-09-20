import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseGenerator(ABC):
    
    def __init__(self, args):
        self.args = args
        self.pipeline = None
    
    @abstractmethod
    def setup(self, model_paths: Dict[str, str]):
        """Setup the generator with model paths."""
        pass
    
    @abstractmethod
    def generate_images(self, prompt: str) -> List[Any]:
        pass
    
    def generate_batch(self, prompts: List[str], model_paths: Dict[str, str], output_dir: Path) -> List[str]:
        self.setup(model_paths)
        
        generated_files = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"Generating prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
            
            images = self.generate_images(prompt)
            
            for img_idx, image in enumerate(images):
                filename = f"prompt_{prompt_idx:03d}_img_{img_idx:02d}_{timestamp}.png"
                filepath = output_dir / filename
                image.save(filepath)
                generated_files.append(str(filepath))
                logger.info(f"Saved: {filename}")
        
        logger.info(f"Generated {len(generated_files)} images")
        return generated_files


class QwenGenerator(BaseGenerator):
    
    def __init__(self, args):
        super().__init__(args)
        self.model_path = None
        self.token_mapping = None
        
        # Configuration
        self.aspect_ratios = {
            "1:1": (1024, 1024), "16:9": (1664, 928), "9:16": (928, 1664),
            "4:3": (1472, 1140), "3:4": (1140, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584)
        }
        self.positive_prompt = "Ultra HD, 4K, cinematic composition."
    
    def setup(self, model_paths: Dict[str, str]):
        logger.info("Setting up Qwen pipeline...")
        
        self.model_paths = model_paths
        self._load_pipeline(model_paths['model'])
        self._load_lora(model_paths.get('lora_weights'))
        self._load_custom_tokenizer(model_paths.get('tokenizer'), model_paths.get('embeddings'))
        self._load_token_mapping(model_paths.get('token_mapping'))
        
        logger.info("Qwen pipeline ready")
    
    def _load_pipeline(self, model_path: str):
        import torch
        from diffusers import QwenImagePipeline
        
        self.model_path = model_path
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)
        
        self.pipeline = QwenImagePipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(device)
        logger.info(f"Loaded pipeline from: {model_path}")
    
    def _load_lora(self, lora_path: str):
        if not lora_path:
            return
        
        import os
        from pathlib import Path
        from safetensors.torch import load_file, save_file
        
        logger.info(f"Loading LoRA: {lora_path}")
        
        lora_file = Path(lora_path)
        cleaned_path = lora_file.parent / f"{lora_file.stem}_cleaned.safetensors"
        
        if not cleaned_path.exists():
            state_dict = load_file(lora_path)
            self._convert_lora_weights(state_dict, cleaned_path)
            logger.info(f"Converted LoRA to: {cleaned_path}")
        
        self.pipeline.load_lora_weights(str(lora_file.parent), weight_name=cleaned_path.name)
        logger.info("LoRA loaded successfully")
    
    def _convert_lora_weights(self, state_dict: dict, output_path: Path):
        from safetensors.torch import save_file
        
        if hasattr(self, 'model_paths') and self.model_paths.get('token_mapping') and "emb_params" in state_dict:
            state_dict.pop("emb_params")
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_state_dict[new_key] = value
        
        save_file(new_state_dict, output_path)
    
    def _load_custom_tokenizer(self, tokenizer_path: str, embeddings_path: str):
        if not (tokenizer_path and embeddings_path):
            return
        
        import torch
        from transformers import AutoTokenizer, Qwen2_5_VLModel
        from safetensors.torch import load_file
        
        logger.info(f"Loading custom tokenizer: {tokenizer_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        device = next(self.pipeline.transformer.parameters()).device
        torch_dtype = next(self.pipeline.transformer.parameters()).dtype
        
        text_encoder = Qwen2_5_VLModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype
        ).to(device)
        
        main_state_dict = text_encoder.language_model.state_dict()
        embeddings_data = load_file(embeddings_path)["emb_params"].to(device)
        
        offset = len(tokenizer.get_vocab()) - embeddings_data.size(0)
        main_state_dict["embed_tokens.weight"] = torch.cat([
            main_state_dict["embed_tokens.weight"][:offset], 
            embeddings_data
        ])
        
        text_encoder.resize_token_embeddings(len(tokenizer))
        text_encoder.language_model.load_state_dict(main_state_dict)
        
        self.pipeline.tokenizer = tokenizer
        self.pipeline.text_encoder = text_encoder
        
        logger.info("Custom tokenizer loaded")
    
    def _load_token_mapping(self, token_mapping_path: str):
        if not token_mapping_path:
            return
        
        import json
        with open(token_mapping_path, 'r') as f:
            self.token_mapping = json.load(f)
        logger.info(f"Loaded token mapping: {len(self.token_mapping)} tokens")
    
    def generate_images(self, prompt: str) -> List[Any]:
        import torch
        
        processed_prompt = self._process_prompt(prompt)
        width, height = self.aspect_ratios[self.args.aspect_ratio]
        
        result = self.pipeline(
            num_images_per_prompt=self.args.num_images_per_prompt,
            prompt=processed_prompt,
            negative_prompt=self.args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=self.args.num_inference_steps,
            true_cfg_scale=self.args.true_cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(self.args.seed)
        )
        
        return result.images
    
    def _process_prompt(self, prompt: str) -> str:
        processed = prompt
        
        if self.token_mapping:
            for token, replacements in self.token_mapping.items():
                if replacements:
                    replacement = replacements[0].replace(" ", "")
                    processed = processed.replace(token, replacement)
        
        return f"{processed} {self.positive_prompt}"


class Generator:
    
    @staticmethod
    def create_generator(model_type: str, args) -> BaseGenerator:
        if model_type == "qwen":
            return QwenGenerator(args)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
