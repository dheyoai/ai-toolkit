import json
import uuid
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional

from services.downloader import Downloader
from services.generator import Generator
from services.cleanup import Cleanup
from utils.logger import setup_logger

logger = setup_logger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    CREATED = "created"
    DOWNLOADING = "downloading"
    GENERATING = "generating"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"


class JobError(Exception):
    def __init__(self, message: str, stage: str = None):
        super().__init__(message)
        self.stage = stage


class JobOrchestrator:    
    def __init__(self, args):
        self.args = args
        self.job_id = str(uuid.uuid4())
        self.job_dir = Path(args.output_dir) / args.job_name / self.job_id
        
        try:
            self.job_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise JobError(f"Failed to create job directory: {str(e)}", "initialization")
        
        self.status = JobStatus.CREATED
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.generated_files = []
        self.error_message = None
        self.error_stage = None
        self.error_traceback = None
        
        try:
            self.downloader = Downloader()
            logger.info("Downloader initialized successfully")
        except Exception as e:
            raise JobError(f"Failed to initialize downloader: {str(e)}", "initialization")
        
        try:
            self.generator = Generator.create_generator(args.model_type, args)
            logger.info(f"Generator for {args.model_type} initialized successfully")
        except Exception as e:
            raise JobError(f"Failed to initialize generator: {str(e)}", "initialization")
        
        self.cleanup = None
        if args.cleanup_local or args.cleanup_cache:
            try:
                self.cleanup = Cleanup(args)
                logger.info("Cleanup service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize cleanup service: {str(e)}")
                # Don't fail the job for cleanup initialization issues
        
        logger.info(f"Job {self.job_id} created successfully")
        self._save_status()
    
    def run_job(self, prompts: List[str]) -> bool:
        try:
            if not prompts:
                raise JobError("No prompts provided", "validation")
            
            logger.info(f"Starting job {self.job_id} with {len(prompts)} prompts")
            
            self._update_status(JobStatus.DOWNLOADING)
            model_paths = self._download_models()
            
            self._update_status(JobStatus.GENERATING)
            self.generated_files = self._generate_images(prompts, model_paths)
            
            if self.cleanup:
                self._update_status(JobStatus.CLEANING)
                self._cleanup_files()
            
            self._update_status(JobStatus.COMPLETED)
            logger.info(f"Job {self.job_id} completed successfully")
            return True
            
        except JobError as e:
            self.error_message = str(e)
            self.error_stage = e.stage
            self.error_traceback = traceback.format_exc()
            self._update_status(JobStatus.FAILED)
            logger.error(f"Job {self.job_id} failed in stage '{e.stage}': {e}")
            return False
            
        except Exception as e:
            self.error_message = f"Unexpected error: {str(e)}"
            self.error_stage = self.status.value
            self.error_traceback = traceback.format_exc()
            self._update_status(JobStatus.FAILED)
            logger.error(f"Job {self.job_id} failed with unexpected error: {e}", exc_info=True)
            return False
    
    def _download_models(self) -> Dict[str, str]:
        try:
            logger.info("Starting model download...")
            model_paths = self.downloader.download_all(self.args)
            
            for component, path in model_paths.items():
                if not Path(path).exists():
                    raise JobError(f"Downloaded {component} path does not exist: {path}", "downloading")
            
            logger.info(f"Successfully downloaded {len(model_paths)} components")
            return model_paths
            
        except Exception as e:
            if isinstance(e, JobError):
                raise
            raise JobError(f"Model download failed: {str(e)}", "downloading")
    
    def _generate_images(self, prompts: List[str], model_paths: Dict[str, str]) -> List[str]:
        try:
            logger.info(f"Starting image generation for {len(prompts)} prompts...")
            
            if not model_paths:
                raise JobError("No model paths provided for generation", "generating")
            
            generated_files = self.generator.generate_batch(prompts, model_paths, self.job_dir)
            
            missing_files = [f for f in generated_files if not Path(f).exists()]
            if missing_files:
                raise JobError(f"Generated files missing: {missing_files[:3]}...", "generating")
            
            logger.info(f"Successfully generated {len(generated_files)} images")
            return generated_files
            
        except Exception as e:
            if isinstance(e, JobError):
                raise
            raise JobError(f"Image generation failed: {str(e)}", "generating")
    
    def _cleanup_files(self):
        try:
            logger.info("Starting cleanup...")
            self.cleanup.cleanup_job(self.generated_files, self.job_dir)
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            # Don't fail the job for cleanup errors, just log them
            logger.warning(f"Cleanup failed (job still successful): {str(e)}")
    
    def _update_status(self, status: JobStatus):
        try:
            self.status = status
            self.updated_at = datetime.now()
            self._save_status()
            logger.info(f"Job {self.job_id} status updated to: {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            # Don't raise here as status updates are not critical
    
    def _save_status(self):
        try:
            status_data = {
                'job_id': self.job_id,
                'job_name': self.args.job_name,
                'status': self.status.value,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'generated_files': self.generated_files,
                'error_message': self.error_message,
                'error_stage': self.error_stage,
                'error_traceback': self.error_traceback
            }
            
            status_file = self.job_dir / "job_status.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save job status: {e}")
            # Don't raise here as status saving is not critical for job execution
    
    def get_status_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'job_name': self.args.job_name,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'generated_files': self.generated_files,
            'error_message': self.error_message,
            'error_stage': self.error_stage,
            'job_dir': str(self.job_dir)
        }