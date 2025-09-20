
from pathlib import Path
from typing import Dict
from huggingface_hub import snapshot_download

from utils.logger import setup_logger

logger = setup_logger(__name__)


class Downloader:
    """Downloader for models and LoRA components."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_all(self, args) -> Dict[str, str]:
        """Download all required models and components."""
        model_paths = {}
        
        model_paths['model'] = self.download_model(args.model_path)
        
        if args.hf_lora_id:
            lora_paths = self.download_lora_bundle(args.hf_lora_id)
            model_paths.update(lora_paths)
        elif args.local_lora_config:
            model_paths.update({
                'lora_weights': args.transformer_lora_path,
                'tokenizer': args.tokenizer_path,
                'embeddings': args.embeddings_path,
                'token_mapping': args.token_abstraction_json_path
            })
        
        logger.info(f"Downloaded components: {list(model_paths.keys())}")
        return model_paths
    
    def download_model(self, model_id: str) -> str:
        """Download model from HF or return local path."""
        if Path(model_id).exists():
            logger.info(f"Using local model: {model_id}")
            return model_id
        
        logger.info(f"Downloading model: {model_id}")
        model_dir = self.cache_dir / "models" / model_id.replace("/", "_")
        
        if model_dir.exists():
            logger.info(f"Model cached: {model_dir}")
            return str(model_dir)
        
        snapshot_download(repo_id=model_id, local_dir=model_dir, local_dir_use_symlinks=False)
        logger.info(f"Downloaded model to: {model_dir}")
        return str(model_dir)
    
    def download_lora_bundle(self, lora_id: str) -> Dict[str, str]:
        """Download LoRA bundle from HuggingFace."""
        logger.info(f"Downloading LoRA bundle: {lora_id}")
        
        lora_dir = self.cache_dir / "lora" / lora_id.replace("/", "_")
        
        if not lora_dir.exists():
            snapshot_download(repo_id=lora_id, local_dir=lora_dir, local_dir_use_symlinks=False)
            logger.info(f"Downloaded LoRA bundle to: {lora_dir}")
        else:
            logger.info(f"LoRA bundle cached: {lora_dir}")
        
        return self._find_lora_components(lora_dir)
    
    def _find_lora_components(self, lora_dir: Path) -> Dict[str, str]:
        """Find LoRA components in downloaded directory."""
        components = {}
        
        safetensors_files = list(lora_dir.glob("*.safetensors"))
        for file in safetensors_files:
            if any(keyword in file.name.lower() for keyword in ['lora', 'adapter']):
                components['lora_weights'] = str(file)
                break
        
        if 'lora_weights' not in components and safetensors_files:
            components['lora_weights'] = str(safetensors_files[0])
        
        if (lora_dir / "tokenizer.json").exists():
            components['tokenizer'] = str(lora_dir)
        else:
            tokenizer_dirs = [d for d in lora_dir.iterdir() 
                            if d.is_dir() and "tokenizer" in d.name.lower()]
            if tokenizer_dirs:
                components['tokenizer'] = str(tokenizer_dirs[0])
        
        for file in safetensors_files:
            if any(keyword in file.name.lower() for keyword in ['embedding', 'emb']):
                components['embeddings'] = str(file)
                break
        
        json_files = list(lora_dir.glob("*.json"))
        for file in json_files:
            if any(keyword in file.name.lower() for keyword in ['token', 'mapping', 'abstraction']):
                components['token_mapping'] = str(file)
                break
        
        logger.info(f"Found components: {list(components.keys())}")
        return components

