from pathlib import Path
from typing import Dict
from huggingface_hub import snapshot_download
import os

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DownloadError(Exception):
    pass


class Downloader:
    
    def __init__(self, cache_dir: str = "/dheyo/lora-infer"):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DownloadError(f"Failed to create cache directory {cache_dir}: {str(e)}")
    
    def download_all(self, args) -> Dict[str, str]:
        try:
            model_paths = {}
            
            logger.info("Downloading base model...")
            model_paths['model'] = self.download_model(args.model_path)
            
            if args.hf_lora_id:
                logger.info("Downloading HuggingFace LoRA bundle...")
                lora_paths = self.download_lora_bundle(args.hf_lora_id)
                model_paths.update(lora_paths)
            elif args.local_lora_config:
                logger.info("Using local LoRA configuration...")
                local_paths = {
                    'lora_weights': args.transformer_lora_path,
                    'tokenizer': args.tokenizer_path,
                    'embeddings': args.embeddings_path,
                    'token_mapping': args.token_abstraction_json_path
                }
                
                for component, path in local_paths.items():
                    if path and not Path(path).exists():
                        raise DownloadError(f"Local {component} path does not exist: {path}")
                
                model_paths.update(local_paths)
            
            logger.info(f"Successfully prepared components: {list(model_paths.keys())}")
            return model_paths
            
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Download process failed: {str(e)}")
    
    def download_model(self, model_id: str) -> str:
        try:
            if Path(model_id).exists():
                logger.info(f"Using local model: {model_id}")
                return model_id
            
            logger.info(f"Downloading model from HuggingFace: {model_id}")
            model_dir = self.cache_dir / "models" / model_id.replace("/", "_")
            
            if model_dir.exists():
                logger.info(f"Model already cached: {model_dir}")
                return str(model_dir)
            
            snapshot_download(
                repo_id=model_id, 
                local_dir=model_dir, 
                local_dir_use_symlinks=False
            )
            
            if not model_dir.exists():
                raise DownloadError(f"Model download completed but directory not found: {model_dir}")
            
            logger.info(f"Successfully downloaded model to: {model_dir}")
            return str(model_dir)
            
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Failed to download model {model_id}: {str(e)}")
    
    def download_lora_bundle(self, lora_id: str) -> Dict[str, str]:
        try:
            logger.info(f"Downloading LoRA bundle: {lora_id}")
            
            lora_dir = self.cache_dir / "lora" / lora_id.replace("/", "_")
            
            if not lora_dir.exists():
                snapshot_download(
                    repo_id=lora_id, 
                    local_dir=lora_dir, 
                    local_dir_use_symlinks=False
                )
                
                if not lora_dir.exists():
                    raise DownloadError(f"LoRA download completed but directory not found: {lora_dir}")
                
                logger.info(f"Successfully downloaded LoRA bundle to: {lora_dir}")
            else:
                logger.info(f"LoRA bundle already cached: {lora_dir}")
            
            return self._find_lora_components(lora_dir)
            
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Failed to download LoRA bundle {lora_id}: {str(e)}")
    
    def _find_lora_components(self, lora_dir: Path) -> Dict[str, str]:
        try:
            components = {}
            
            safetensors_files = list(lora_dir.glob("*.safetensors"))
            if not safetensors_files:
                raise DownloadError(f"No .safetensors files found in LoRA directory: {lora_dir}")
            
            for file in safetensors_files:
                if any(keyword in file.name.lower() for keyword in ['lora', 'adapter']):
                    components['lora_weights'] = str(file)
                    break
            
            if 'lora_weights' not in components:
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
            
            if not components:
                raise DownloadError(f"No valid LoRA components found in directory: {lora_dir}")
            
            logger.info(f"Found LoRA components: {list(components.keys())}")
            return components
            
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Failed to find LoRA components: {str(e)}")
