import uuid
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List

from services.redis_processor import RedisProcessor
from services.downloader import Downloader
from services.generator import Generator
from services.uploader import Uploader
from services.cleanup import Cleanup
from utils.logger import setup_logger

logger = setup_logger(__name__)


class JobStatus(Enum):
    CREATED = "created"
    DOWNLOADING = "downloading"
    GENERATING = "generating"
    UPLOADING = "uploading"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"


class JobOrchestrator:
    def __init__(self, request_queue: str, response_queue: str, timeout: int = 30, uploader_type: str = "s3"):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout = timeout
        self.uploader_type = uploader_type
        
        self.redis_processor = RedisProcessor()
        
    def run(self):
        logger.info(f"Starting job orchestrator with queues: {self.request_queue} -> {self.response_queue}")
        logger.info(f"Will timeout after {self.timeout}s if no jobs available")
        
        job_count = 0
        while True:
            try:
                logger.info(f"Waiting for job requests... (timeout: {self.timeout}s)")
                job_request = self.redis_processor.pop_request(self.request_queue, timeout=self.timeout)
                
                if job_request is None:
                    logger.info(f"No jobs received within {self.timeout}s timeout. Processed {job_count} jobs total. Exiting.")
                    break
                
                job_count += 1
                logger.info(f"Received job {job_count}, processing...")
                self._process_job(job_request, job_count)
                
            except KeyboardInterrupt:
                logger.info(f"Consumer interrupted by user. Processed {job_count} jobs total.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                continue
        
        logger.info(f"Job orchestrator finished. Total jobs processed: {job_count}")
        return job_count
    
    def _process_job(self, job_request: Dict[str, Any], job_number: int):
        job_id = job_request.get("generation_id")
        
        job_state = {
            "job_id": job_id,
            "job_number": job_number,
            "status": JobStatus.CREATED.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "stages": {
                "download": {"status": "pending", "start_time": None, "end_time": None, "error": None, "details": {}},
                "generate": {"status": "pending", "start_time": None, "end_time": None, "error": None, "details": {}},
                "upload": {"status": "pending", "start_time": None, "end_time": None, "error": None, "details": {}},
                "cleanup": {"status": "pending", "start_time": None, "end_time": None, "error": None, "details": {}}
            },
            "generated_files": [],
            "uploaded_files": [],
            "upload_details": [],
            "error_message": None,
            "error_stage": None,
            "processing_time": None,
            "request_details": {
                "model_type": job_request.get("model_type"),
                "model_path": job_request.get("model_path"),
                "num_images": job_request.get("num_images_per_prompt", 1),
                "inference_steps": job_request.get("num_inference_steps", 50),
                "aspect_ratio": job_request.get("aspect_ratio", "16:9"),
                "seed": job_request.get("seed", 42),
                "has_lora": bool(job_request.get("hf_lora_id") or job_request.get("local_lora_directory") or 
                               (job_request.get("transformer_lora_path") and job_request.get("tokenizer_path"))),
                "cleanup_requested": {
                    "local": job_request.get("cleanup_local", False),
                    "cache": job_request.get("cleanup_cache", False)
                }
            }
        }
        
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Processing job #{job_number} with ID: {job_id}")
            
            self._update_job_status(job_state, JobStatus.DOWNLOADING)
            model_paths = self._download_stage(job_request, job_state)
            
            self._update_job_status(job_state, JobStatus.GENERATING)
            generated_files = self._generate_stage(job_request, model_paths, job_state)
            
            self._update_job_status(job_state, JobStatus.UPLOADING)
            uploaded_files, upload_details = self._upload_stage(job_request, generated_files, job_id, job_state)
            
            self._update_job_status(job_state, JobStatus.CLEANING)
            self._cleanup_stage(job_request, generated_files, job_state)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            job_state.update({
                "status": JobStatus.COMPLETED.value,
                "success": True,
                "generated_files": generated_files,
                "uploaded_files": uploaded_files,
                "upload_details": upload_details,
                "processing_time": round(processing_time, 2),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"Job #{job_number} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Job #{job_number} failed after {processing_time:.2f}s: {error_msg}")
            
            job_state.update({
                "status": JobStatus.FAILED.value,
                "error_message": error_msg,
                "error_stage": job_state["status"],
                "processing_time": round(processing_time, 2),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        
        finally:
            self._save_job_status(job_state)
            self._push_response(job_state, job_request)
    
    def _update_job_status(self, job_state: Dict[str, Any], status: JobStatus):
        job_state["status"] = status.value
        job_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Job {job_state['job_id']} status: {status.value}")
    
    def _download_stage(self, job_request: Dict[str, Any], job_state: Dict[str, Any]) -> Dict[str, str]:
        stage = job_state["stages"]["download"]
        stage["start_time"] = datetime.now(timezone.utc).isoformat()
        stage["status"] = "running"
        
        try:
            downloader = Downloader()
            model_paths = downloader.download_all(job_request)
            
            stage.update({
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "components_downloaded": list(model_paths.keys()),
                    "model_path": model_paths.get("model"),
                    "has_lora": "lora_weights" in model_paths,
                    "has_custom_tokenizer": "tokenizer" in model_paths,
                    "has_embeddings": "embeddings" in model_paths
                }
            })
            
            logger.info(f"Download completed: {len(model_paths)} components")
            return model_paths
            
        except Exception as e:
            stage.update({
                "status": "failed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            })
            raise
    
    def _generate_stage(self, job_request: Dict[str, Any], model_paths: Dict[str, str], job_state: Dict[str, Any]) -> List[str]:
        stage = job_state["stages"]["generate"]
        stage["start_time"] = datetime.now(timezone.utc).isoformat()
        stage["status"] = "running"
        
        try:
            generator = Generator.create_generator(job_request)
            generated_files = generator.generate_batch(job_request, model_paths)
            
            stage.update({
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "files_generated": len(generated_files),
                    "images_per_prompt": job_request.get("num_images_per_prompt", 1),
                    "inference_steps": job_request.get("num_inference_steps", 50),
                    "model_type": job_request.get("model_type"),
                    "aspect_ratio": job_request.get("aspect_ratio")
                }
            })
            
            logger.info(f"Generation completed: {len(generated_files)} files")
            return generated_files
            
        except Exception as e:
            stage.update({
                "status": "failed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            })
            raise
    
    def _upload_stage(self, job_request: Dict[str, Any], generated_files: List[str], job_id: str, job_state: Dict[str, Any]) -> tuple:
        stage = job_state["stages"]["upload"]
        stage["start_time"] = datetime.now(timezone.utc).isoformat()
        stage["status"] = "running"
        
        try:
            if not generated_files:
                stage.update({
                    "status": "skipped",
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "details": {"reason": "no_files_to_upload"}
                })
                return [], []
            
            uploader = Uploader.create_uploader(self.uploader_type)
            user_id = f"{job_request.get('user_id')}"
            
            upload_results = uploader.upload_generated_files(
                generated_files=generated_files,
                user_id=user_id,
                generation_id=job_id
            )
            
            successful_uploads = [r for r in upload_results if r['success']]
            failed_uploads = [r for r in upload_results if not r['success']]
            uploaded_files = [r['s3_url'] for r in successful_uploads]
            
            stage.update({
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "total_files": len(generated_files),
                    "successful_uploads": len(successful_uploads),
                    "failed_uploads": len(failed_uploads),
                    "upload_success_rate": round(len(successful_uploads) / len(generated_files) * 100, 2) if generated_files else 0,
                    "user_id": user_id,
                    "generation_id": job_id,
                    "failed_files": [r['local_path'] for r in failed_uploads] if failed_uploads else []
                }
            })
            
            logger.info(f"Upload completed: {len(successful_uploads)}/{len(generated_files)} files")
            return uploaded_files, upload_results
            
        except Exception as e:
            stage.update({
                "status": "failed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            })
            raise
    
    def _cleanup_stage(self, job_request: Dict[str, Any], generated_files: List[str], job_state: Dict[str, Any]):
        stage = job_state["stages"]["cleanup"]
        stage["start_time"] = datetime.now(timezone.utc).isoformat()
        stage["status"] = "running"
        
        try:
            cleanup_local = job_request.get("cleanup_local", False)
            cleanup_cache = job_request.get("cleanup_cache", False)
            
            if not cleanup_local and not cleanup_cache:
                stage.update({
                    "status": "skipped",
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "details": {"reason": "cleanup_not_requested"}
                })
                return
            
            cleanup = Cleanup()
            cleanup.cleanup_job(job_request, generated_files)
            
            stage.update({
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "cleanup_local": cleanup_local,
                    "cleanup_cache": cleanup_cache,
                    "files_cleaned": len(generated_files) if cleanup_local else 0
                }
            })
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            stage.update({
                "status": "failed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            })
            logger.warning(f"Cleanup failed (non-critical): {str(e)}")
    
    def _save_job_status(self, job_state: Dict[str, Any]):
        try:
            output_dir = Path("./job_status")
            output_dir.mkdir(exist_ok=True)
            
            status_file = output_dir / f"job_{job_state['job_id']}.json"
            with open(status_file, 'w') as f:
                json.dump(job_state, f, indent=2)
                
            logger.debug(f"Job status saved to: {status_file}")
            
        except Exception as e:
            logger.error(f"Failed to save job status: {str(e)}")
    
    def _push_response(self, job_state: Dict[str, Any], job_request: Dict[str, Any]):
        try:
            public_url = None
            uploaded_files = job_state.get("uploaded_files", [])
            if uploaded_files:
                public_url = uploaded_files[0]
            
            model_info = job_request.get("model_path", "unknown")
            if job_request.get("hf_lora_id"):
                model_info = f"{model_info} + {job_request['hf_lora_id']}"
            
            generation_details = job_state["stages"]["generate"]["details"]
            aspect_ratio = job_request.get("aspect_ratio", "16:9")
            
            aspect_ratios = {
                "1:1": (1024, 1024), "16:9": (1664, 928), "9:16": (928, 1664),
                "4:3": (1472, 1140), "3:4": (1140, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584)
            }
            width, height = aspect_ratios.get(aspect_ratio, (1024, 1024))
            
            response = {
                "generation_id": job_state["job_id"],
                "status": "completed" if job_state["success"] else "failed",
                "timestamp": job_state["updated_at"],
                "agent": "qwen_consumer",
                "error_message": job_state.get("error_message"),
                "public_url": public_url,
                "external_id": job_state["job_id"],
                "metadata": {
                    "user_id": job_request.get("user_id"),
                    "model": model_info,
                    "prompt": job_request.get("instruction"),
                    "generation_type": "text-to-image",
                    "width": width,
                    "height": height,
                    "file_type": "image/png",
                    "filename": f"generated_image_{job_state['job_id']}.png"
                }
            }
            
            if job_state["success"]:
                images = []
                for i, url in enumerate(uploaded_files):
                    images.append({
                        "url": url,
                        "content_type": "image/png",
                        "file_size": None
                    })
                
                response["response"] = {
                    "request_id": job_state["job_id"],
                    "status": "completed",
                    "images": images,
                    "metrics": {
                        "duration": job_state.get("processing_time"),
                        "seed": job_request.get("seed", 42),
                        "steps": job_request.get("num_inference_steps", 50)
                    }
                }
            else:
                response["response"] = {
                    "error": {
                        "code": "GENERATION_FAILED",
                        "message": job_state.get("error_message", "Unknown error occurred"),
                        "details": {
                            "error_stage": job_state.get("error_stage"),
                            "processing_time": job_state.get("processing_time")
                        }
                    }
                }
            
            self.redis_processor.push_response(self.response_queue, response)
            
        except Exception as e:
            logger.error(f"Failed to push response: {str(e)}")