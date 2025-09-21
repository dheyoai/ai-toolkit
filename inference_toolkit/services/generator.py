import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GeneratorError(Exception):
    pass


class BaseGenerator(ABC):
    def __init__(self, job_request: Dict[str, Any]):
        self.job_request = job_request
        self.pipeline = None
        self.is_setup = False
        self._validate_request()
    
    def _validate_request(self):
        required_fields = [
            "model_type", "instruction", "num_images_per_prompt",
            "num_inference_steps", "true_cfg_scale", "aspect_ratio",
            "seed", "dtype", "negative_prompt", "output_dir"
        ]
        
        missing = [field for field in required_fields if field not in self.job_request]
        if missing:
            raise GeneratorError(f"Missing required fields: {missing}")
    
    @abstractmethod
    def setup(self, model_paths: Dict[str, str]):
        pass
    
    @abstractmethod
    def generate_images(self, prompt: str) -> List[Any]:
        pass
    
    def generate_batch(self, job_request: Dict[str, Any], model_paths: Dict[str, str]) -> List[str]:
        try:
            if not model_paths:
                raise GeneratorError("No model paths provided for generation")
            
            prompts = self._get_prompts(job_request)
            output_dir = self._create_output_dir(job_request)
            
            logger.info("Setting up generation pipeline...")
            self.setup(model_paths)
            
            generated_files = []
            
            logger.info(f"Starting batch generation for {len(prompts)} prompts")
            
            for prompt_idx, prompt in enumerate(prompts):
                try:
                    logger.info(f"Generating prompt {prompt_idx + 1}/{len(prompts)}")
                    
                    if not prompt or not prompt.strip():
                        logger.warning(f"Empty prompt at index {prompt_idx}, skipping")
                        continue
                    
                    images = self.generate_images(prompt.strip())
                    
                    if not images:
                        logger.warning(f"No images generated for prompt {prompt_idx}")
                        continue

                    for img_idx, image in enumerate(images):
                        try:
                            image_uuid = str(uuid.uuid4())
                            filename = f"output_{image_uuid}.png"
                            filepath = output_dir / filename
                            
                            if hasattr(image, 'save'):
                                image.save(filepath)
                                
                                if filepath.exists() and filepath.stat().st_size > 0:
                                    generated_files.append(str(filepath))
                                    logger.info(f"Saved: {filename}")
                                else:
                                    logger.error(f"Failed to save image properly: {filename}")
                            else:
                                raise GeneratorError(f"Invalid image object for prompt {prompt_idx}, image {img_idx}")
                                
                        except Exception as e:
                            logger.error(f"Failed to save image {img_idx} for prompt {prompt_idx}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Failed to generate images for prompt {prompt_idx}: {str(e)}")
                    continue
            
            if not generated_files:
                raise GeneratorError("No images were successfully generated")
            
            logger.info(f"Successfully generated {len(generated_files)} images")
            return generated_files
            
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Batch generation failed: {str(e)}")
    
    def _get_prompts(self, job_request: Dict[str, Any]) -> List[str]:
        if job_request.get("prompts_path"):
            prompts_file = Path(job_request["prompts_path"])
            if not prompts_file.exists():
                raise GeneratorError(f"Prompts file does not exist: {prompts_file}")
            return prompts_file.read_text().strip().splitlines()
        elif job_request.get("prompts"):
            return job_request["prompts"]
        elif job_request.get("instruction"):
            return [job_request["instruction"]]
        else:
            raise GeneratorError("No prompts provided (instruction, prompts, or prompts_path)")
    
    def _create_output_dir(self, job_request: Dict[str, Any]) -> Path:
        output_dir = Path(job_request["output_dir"]) / job_request.get("job_name", "unknown") / str(uuid.uuid4())
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        except Exception as e:
            raise GeneratorError(f"Failed to create output directory {output_dir}: {str(e)}")


class QwenGenerator(BaseGenerator):
    def __init__(self, job_request: Dict[str, Any]):
        super().__init__(job_request)
        self.model_path = None
        self.token_mapping = None
        
        valid_dtypes = ["fp32", "fp16", "bf16"]
        if job_request["dtype"] not in valid_dtypes:
            raise GeneratorError(f"Invalid dtype '{job_request['dtype']}'. Must be one of: {valid_dtypes}")
        
        self.aspect_ratios = {
            "1:1": (1024, 1024), "16:9": (1664, 928), "9:16": (928, 1664),
            "4:3": (1472, 1140), "3:4": (1140, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584)
        }
        
        if job_request["aspect_ratio"] not in self.aspect_ratios:
            raise GeneratorError(f"Invalid aspect ratio '{job_request['aspect_ratio']}'. Must be one of: {list(self.aspect_ratios.keys())}")
        
        self.positive_prompt = "Ultra HD, 4K, cinematic composition."
        logger.info("QwenGenerator initialized successfully")
    
    def setup(self, model_paths: Dict[str, str]):
        if self.is_setup:
            logger.info("Pipeline already setup, skipping...")
            return
        
        try:
            logger.info("Setting up Qwen pipeline...")
            
            model_path = model_paths.get('model')
            if not model_path:
                raise GeneratorError("No model path provided in model_paths")
            
            if not Path(model_path).exists():
                raise GeneratorError(f"Model path does not exist: {model_path}")
            
            self.model_paths = model_paths
            
            self._load_pipeline(model_path)
            
            lora_path = model_paths.get('lora_weights')
            if lora_path:
                self._load_lora(lora_path)
            
            tokenizer_path = model_paths.get('tokenizer')
            embeddings_path = model_paths.get('embeddings')
            if tokenizer_path and embeddings_path:
                self._load_custom_tokenizer(tokenizer_path, embeddings_path)
            
            token_mapping_path = model_paths.get('token_mapping')
            if token_mapping_path:
                self._load_token_mapping(token_mapping_path)
            
            self.is_setup = True
            logger.info("Qwen pipeline setup completed successfully")
            
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Pipeline setup failed: {str(e)}")
    
    def _load_pipeline(self, model_path: str):
        try:
            import torch
            from diffusers import QwenImagePipeline
            
            self.model_path = model_path
            
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU (will be very slow)")
                device = "cpu"
            else:
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            
            dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
            torch_dtype = dtype_map[self.job_request["dtype"]]
            
            logger.info(f"Loading pipeline with dtype: {self.job_request['dtype']}")

            try:
                self.pipeline = QwenImagePipeline.from_pretrained(
                    model_path, 
                    torch_dtype=torch_dtype
                ).to(device)
            except Exception as e:
                raise GeneratorError(f"Failed to load QwenImagePipeline from {model_path}: {str(e)}")

            if self.pipeline is None:
                raise GeneratorError("Pipeline loaded but is None")
            
            logger.info(f"Successfully loaded pipeline from: {model_path}")
            
        except ImportError as e:
            raise GeneratorError(f"Failed to import required libraries: {str(e)}")
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Failed to load pipeline: {str(e)}")
    
    def _load_lora(self, lora_path: str):
        try:
            if not lora_path:
                return
            
            logger.info(f"Loading LoRA: {lora_path}")
            
            lora_file = Path(lora_path)
            if not lora_file.exists():
                raise GeneratorError(f"LoRA file does not exist: {lora_path}")
            
            if not lora_file.suffix == '.safetensors':
                raise GeneratorError(f"LoRA file must be .safetensors format: {lora_path}")
            
            try:
                from safetensors.torch import load_file, save_file
            except ImportError as e:
                raise GeneratorError(f"Failed to import safetensors: {str(e)}")
            
            cleaned_path = lora_file.parent / f"{lora_file.stem}_cleaned.safetensors"
            
            if not cleaned_path.exists():
                try:
                    state_dict = load_file(lora_path)
                    self._convert_lora_weights(state_dict, cleaned_path)
                    logger.info(f"Converted LoRA to: {cleaned_path}")
                except Exception as e:
                    raise GeneratorError(f"Failed to convert LoRA weights: {str(e)}")

            try:
                self.pipeline.load_lora_weights(str(lora_file.parent), weight_name=cleaned_path.name)
                logger.info("LoRA loaded successfully")
            except Exception as e:
                raise GeneratorError(f"Failed to load LoRA into pipeline: {str(e)}")
                
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"LoRA loading failed: {str(e)}")
    
    def _convert_lora_weights(self, state_dict: dict, output_path: Path):
        try:
            from safetensors.torch import save_file
            
            if hasattr(self, 'model_paths') and self.model_paths.get('token_mapping') and "emb_params" in state_dict:
                state_dict.pop("emb_params")
                logger.info("Removed emb_params from LoRA weights")

            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("diffusion_model.", "transformer.")
                new_state_dict[new_key] = value

            save_file(new_state_dict, output_path)
            logger.info(f"Converted {len(new_state_dict)} LoRA weight tensors")
            
        except Exception as e:
            raise GeneratorError(f"Failed to convert LoRA weights: {str(e)}")
    
    def _load_custom_tokenizer(self, tokenizer_path: str, embeddings_path: str):
        try:
            if not (tokenizer_path and embeddings_path):
                return
            
            logger.info(f"Loading custom tokenizer: {tokenizer_path}")

            if not Path(tokenizer_path).exists():
                raise GeneratorError(f"Tokenizer path does not exist: {tokenizer_path}")
            
            if not Path(embeddings_path).exists():
                raise GeneratorError(f"Embeddings path does not exist: {embeddings_path}")
            
            try:
                import torch
                from transformers import AutoTokenizer, Qwen2_5_VLModel
                from safetensors.torch import load_file
            except ImportError as e:
                raise GeneratorError(f"Failed to import required libraries for tokenizer: {str(e)}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                raise GeneratorError(f"Failed to load tokenizer from {tokenizer_path}: {str(e)}")

            device = next(self.pipeline.transformer.parameters()).device
            torch_dtype = next(self.pipeline.transformer.parameters()).dtype

            try:
                text_encoder = Qwen2_5_VLModel.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder",
                    torch_dtype=torch_dtype
                ).to(device)
            except Exception as e:
                raise GeneratorError(f"Failed to load text encoder: {str(e)}")

            try:
                embeddings_data = load_file(embeddings_path)["emb_params"].to(device)
            except KeyError:
                raise GeneratorError(f"No 'emb_params' found in embeddings file: {embeddings_path}")
            except Exception as e:
                raise GeneratorError(f"Failed to load embeddings from {embeddings_path}: {str(e)}")
            
            try:
                main_state_dict = text_encoder.language_model.state_dict()
                offset = len(tokenizer.get_vocab()) - embeddings_data.size(0)
                main_state_dict["embed_tokens.weight"] = torch.cat([
                    main_state_dict["embed_tokens.weight"][:offset], 
                    embeddings_data
                ])
                
                text_encoder.resize_token_embeddings(len(tokenizer))
                text_encoder.language_model.load_state_dict(main_state_dict)
                
                self.pipeline.tokenizer = tokenizer
                self.pipeline.text_encoder = text_encoder
                
                logger.info("Custom tokenizer and embeddings loaded successfully")
                
            except Exception as e:
                raise GeneratorError(f"Failed to update embeddings: {str(e)}")
                
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Custom tokenizer loading failed: {str(e)}")
    
    def _load_token_mapping(self, token_mapping_path: str):
        try:
            if not token_mapping_path:
                return
            
            logger.info(f"Loading token mapping: {token_mapping_path}")

            mapping_file = Path(token_mapping_path)
            if not mapping_file.exists():
                raise GeneratorError(f"Token mapping file does not exist: {token_mapping_path}")

            try:
                import json
                with open(token_mapping_path, 'r') as f:
                    self.token_mapping = json.load(f)
                    
                if not isinstance(self.token_mapping, dict):
                    raise GeneratorError(f"Token mapping must be a dictionary, got: {type(self.token_mapping)}")
                
                logger.info(f"Loaded token mapping with {len(self.token_mapping)} tokens")
                
            except json.JSONDecodeError as e:
                raise GeneratorError(f"Invalid JSON in token mapping file: {str(e)}")
            except Exception as e:
                raise GeneratorError(f"Failed to load token mapping: {str(e)}")
                
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Token mapping loading failed: {str(e)}")
    
    def generate_images(self, prompt: str) -> List[Any]:
        try:
            if not self.is_setup:
                raise GeneratorError("Pipeline not set up. Call setup() first.")
            
            if not prompt or not prompt.strip():
                raise GeneratorError("Empty prompt provided")

            processed_prompt = self._process_prompt(prompt)
            width, height = self.aspect_ratios[self.job_request["aspect_ratio"]]
            
            logger.info(f"Generating {self.job_request['num_images_per_prompt']} images at {width}x{height}")

            try:
                import torch
            except ImportError as e:
                raise GeneratorError(f"Failed to import torch: {str(e)}")

            try:
                result = self.pipeline(
                    num_images_per_prompt=self.job_request["num_images_per_prompt"],
                    prompt=processed_prompt,
                    negative_prompt=self.job_request["negative_prompt"],
                    width=width,
                    height=height,
                    num_inference_steps=self.job_request["num_inference_steps"],
                    true_cfg_scale=self.job_request["true_cfg_scale"],
                    generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(self.job_request["seed"])
                )
                
                if not hasattr(result, 'images') or not result.images:
                    raise GeneratorError("Pipeline returned no images")
                
                logger.info(f"Successfully generated {len(result.images)} images")
                return result.images
                
            except Exception as e:
                raise GeneratorError(f"Image generation failed: {str(e)}")
            
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Unexpected error during generation: {str(e)}")
    
    def _process_prompt(self, prompt: str) -> str:
        try:
            processed = prompt.strip()

            if self.token_mapping:
                for token, replacements in self.token_mapping.items():
                    if replacements and isinstance(replacements, list) and replacements[0]:
                        replacement = replacements[0].replace(" ", "")
                        processed = processed.replace(token, replacement)
            
            final_prompt = f"{processed} {self.positive_prompt}"
            
            logger.debug(f"Processed prompt: {final_prompt}")
            return final_prompt
            
        except Exception as e:
            logger.warning(f"Error processing prompt, using original: {str(e)}")
            return f"{prompt} {self.positive_prompt}"


class Generator:
    @staticmethod
    def create_generator(job_request: Dict[str, Any]) -> BaseGenerator:
        try:
            model_type = job_request.get("model_type")
            if model_type == "qwen":
                return QwenGenerator(job_request)
            else:
                raise GeneratorError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            if isinstance(e, GeneratorError):
                raise
            raise GeneratorError(f"Failed to create generator: {str(e)}")
