from pathlib import Path
from typing import Dict
from huggingface_hub import snapshot_download
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DownloadError(Exception):
    pass


class Downloader:
    def __init__(self):
        pass
    
    def download_all(self, job_request: Dict[str, str]) -> Dict[str, str]:
        try:
            self._validate_request(job_request)
            
            cache_dir = job_request.get("cache_dir", "./cache")
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            model_paths = {}
            
            logger.info("Downloading base model...")
            model_paths['model'] = self.download_model(job_request["model_path"])
            
            if job_request.get("hf_lora_id"):
                logger.info("Downloading HuggingFace LoRA bundle...")
                lora_paths = self.download_lora_bundle(job_request["hf_lora_id"])
                model_paths.update(lora_paths)
            elif job_request.get("local_lora_directory"):
                logger.info("Using local LoRA directory...")
                lora_paths = self._get_lora_from_directory(job_request["local_lora_directory"])
                model_paths.update(lora_paths)
            elif job_request.get("local_lora_config"):
                logger.info("Using local LoRA configuration...")
                local_paths = self._get_local_lora_paths(job_request)
                model_paths.update(local_paths)
            
            logger.info(f"Successfully prepared components: {list(model_paths.keys())}")
            return model_paths
            
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Download process failed: {str(e)}")
    
    def _validate_request(self, job_request: Dict[str, str]):
        if not job_request.get("model_path"):
            raise DownloadError("model_path is required")
        
        if job_request.get("local_lora_config"):
            required_fields = ["transformer_lora_path", "tokenizer_path", "embeddings_path"]
            missing = [field for field in required_fields if not job_request.get(field)]
            if missing:
                raise DownloadError(f"local_lora_config requires: {missing}")
            
            for field in required_fields:
                path = job_request[field]
                if path and not Path(path).exists():
                    raise DownloadError(f"Local {field} path does not exist: {path}")
        
        if job_request.get("local_lora_directory"):
            lora_dir = Path(job_request["local_lora_directory"])
            if not lora_dir.exists():
                raise DownloadError(f"Local LoRA directory does not exist: {lora_dir}")
            if not lora_dir.is_dir():
                raise DownloadError(f"Local LoRA path is not a directory: {lora_dir}")
    
    def _get_lora_from_directory(self, lora_directory: str) -> Dict[str, str]:
        lora_dir = Path(lora_directory)
        logger.info(f"Finding LoRA components in directory: {lora_dir}")
        return self._find_lora_components(lora_dir)
    
    def _get_local_lora_paths(self, job_request: Dict[str, str]) -> Dict[str, str]:
        return {
            'lora_weights': job_request.get('transformer_lora_path'),
            'tokenizer': job_request.get('tokenizer_path'),
            'embeddings': job_request.get('embeddings_path'),
            'token_mapping': job_request.get('token_abstraction_json_path')
        }
    
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
            
            # Find tokenizer directory
            tokenizer_found = False
            for item in lora_dir.iterdir():
                if item.is_dir() and "tokenizer" in item.name:
                    components['tokenizer'] = str(item)
                    logger.info(f"Found tokenizer: {item.name}")
                    tokenizer_found = True
                    break
            
            if not tokenizer_found:
                raise DownloadError(f"Tokenizer directory not found in {lora_dir}")
            
            # Find LoRA weights
            lora_weights_found = False
            for item in lora_dir.iterdir():
                if item.is_file() and "LoRA" in item.name and item.suffix == ".safetensors":
                    components['lora_weights'] = str(item)
                    logger.info(f"Found LoRA weights: {item.name}")
                    lora_weights_found = True
                    break
            
            if not lora_weights_found:
                raise DownloadError(f"LoRA weights file (containing 'LoRA' and ending with '.safetensors') not found in {lora_dir}")
            
            # Find embeddings
            embeddings_found = False
            for item in lora_dir.iterdir():
                if item.is_file() and "embeddings" in item.name and item.suffix == ".safetensors":
                    components['embeddings'] = str(item)
                    logger.info(f"Found embeddings: {item.name}")
                    embeddings_found = True
                    break
            
            if not embeddings_found:
                raise DownloadError(f"Embeddings file (containing 'embeddings' and ending with '.safetensors') not found in {lora_dir}")
            
            # Find token mapping
            tokens_file = lora_dir / "tokens.json"
            if tokens_file.exists():
                components['token_mapping'] = str(tokens_file)
                logger.info(f"Found token mapping: tokens.json")
            else:
                raise DownloadError(f"Token mapping file 'tokens.json' not found in {lora_dir}")
            
            logger.info(f"Found all LoRA components: {list(components.keys())}")
            return components
            
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            raise DownloadError(f"Failed to find LoRA components: {str(e)}")
