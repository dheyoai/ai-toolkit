import json
import redis
import os
from typing import Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RedisProcessorError(Exception):
    pass


class RequestSchema:
    REQUIRED_FIELDS = [
        "generation_id", "user_id", "model_type", "model_path", "instruction",
        "num_images_per_prompt", "num_inference_steps", "true_cfg_scale",
        "aspect_ratio", "seed", "dtype", "negative_prompt", "output_dir"
    ]
    
    OPTIONAL_FIELDS = [
        "hf_lora_id", "local_lora_directory", "local_lora_config",
        "transformer_lora_path", "tokenizer_path", "embeddings_path", 
        "token_abstraction_json_path", "cache_dir", "cleanup_local", 
        "cleanup_cache", "prompts", "prompts_path"
    ]
    
    @staticmethod
    def validate(request_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request_dict, dict):
            raise RedisProcessorError("Request must be a JSON object")
        
        missing_fields = [field for field in RequestSchema.REQUIRED_FIELDS 
                         if field not in request_dict]
        if missing_fields:
            raise RedisProcessorError(f"Missing required fields: {missing_fields}")
        
        validated = {}
        for field in RequestSchema.REQUIRED_FIELDS + RequestSchema.OPTIONAL_FIELDS:
            if field in request_dict:
                validated[field] = request_dict[field]
        
        validated.setdefault("cache_dir", os.getenv("DEFAULT_CACHE_DIR", "./cache"))
        validated.setdefault("cleanup_local", False)
        validated.setdefault("cleanup_cache", False)
        validated.setdefault("local_lora_config", False)
        
        return validated


class RedisProcessor:
    def __init__(self):
        self._validate_environment()
        self._initialize_client()
    
    def _validate_environment(self):
        self.redis_host = os.getenv('REDIS_HOST')
        
        try:
            self.redis_port = int(os.getenv('REDIS_PORT'))
            self.redis_db = int(os.getenv('REDIS_DB'))
        except (ValueError, TypeError) as e:
            raise RedisProcessorError(f"Invalid Redis port or database number: {str(e)}")
        
        redis_password = os.getenv('REDIS_PASSWORD')
        self.redis_password = redis_password if redis_password else None
        
        if self.redis_port <= 0 or self.redis_port > 65535:
            raise RedisProcessorError(f"Invalid Redis port: {self.redis_port}. Must be between 1-65535")
        
        if self.redis_db < 0:
            raise RedisProcessorError(f"Invalid Redis database number: {self.redis_db}. Must be >= 0")
    
    def _initialize_client(self):
        try:
            connection_params = {
                'host': self.redis_host,
                'port': self.redis_port,
                'db': self.redis_db,
                'decode_responses': True
            }
            
            if self.redis_password:
                connection_params['password'] = self.redis_password
            
            self.redis_client = redis.Redis(**connection_params)
            
            self._test_connection()
            
            logger.info(f"Redis processor initialized for {self.redis_host}:{self.redis_port}/{self.redis_db}")
            
        except Exception as e:
            raise RedisProcessorError(f"Failed to initialize Redis client: {str(e)}")
    
    def _test_connection(self):
        try:
            self.redis_client.ping()
            logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
            
        except redis.ConnectionError as e:
            raise RedisProcessorError(f"Cannot connect to Redis server at {self.redis_host}:{self.redis_port}: {str(e)}")
        except redis.AuthenticationError as e:
            raise RedisProcessorError(f"Redis authentication failed: {str(e)}")
        except Exception as e:
            raise RedisProcessorError(f"Redis connection test failed: {str(e)}")
    
    def pop_request(self, request_queue: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        try:
            if not request_queue or not isinstance(request_queue, str):
                raise RedisProcessorError("Request queue name must be a non-empty string")
            
            result = self.redis_client.blpop(request_queue, timeout=timeout)
            
            if result is None:
                return None
            
            queue_name, request_data = result
            logger.info(f"Received request from queue: {queue_name}")
            
            try:
                request_dict = json.loads(request_data)
            except json.JSONDecodeError as e:
                raise RedisProcessorError(f"Invalid JSON in request: {str(e)}")
            
            validated_request = RequestSchema.validate(request_dict)
            logger.info(f"Successfully parsed request for generation: {validated_request['generation_id']}")
            
            return validated_request
            
        except RedisProcessorError:
            raise
        except Exception as e:
            raise RedisProcessorError(f"Failed to pop request: {str(e)}")
    
    def push_response(self, response_queue: str, job_result: Dict[str, Any]) -> bool:
        try:
            if not response_queue or not isinstance(response_queue, str):
                raise RedisProcessorError("Response queue name must be a non-empty string")
            
            if not isinstance(job_result, dict):
                raise RedisProcessorError("Job result must be a dictionary")
            
            try:
                response_data = json.dumps(job_result, indent=2, default=str)
            except TypeError as e:
                raise RedisProcessorError(f"Failed to serialize response: {str(e)}")
            
            queue_length = self.redis_client.rpush(response_queue, response_data)
            logger.info(f"Pushed response to queue: {response_queue} (length: {queue_length})")
            
            return True
            
        except RedisProcessorError:
            raise
        except Exception as e:
            raise RedisProcessorError(f"Failed to push response: {str(e)}")
    
    def get_queue_length(self, queue_name: str) -> int:
        try:
            if not queue_name or not isinstance(queue_name, str):
                logger.error("Queue name must be a non-empty string")
                return 0
            
            return self.redis_client.llen(queue_name)
        except Exception as e:
            logger.error(f"Failed to get queue length for {queue_name}: {str(e)}")
            return 0
    
    def clear_queue(self, queue_name: str) -> bool:
        try:
            if not queue_name or not isinstance(queue_name, str):
                logger.error("Queue name must be a non-empty string")
                return False
            
            self.redis_client.delete(queue_name)
            logger.info(f"Cleared queue: {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear queue {queue_name}: {str(e)}")
            return False
            