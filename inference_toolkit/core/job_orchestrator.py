
import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any

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


class JobOrchestrator:    
    def __init__(self, args):
        self.args = args
        self.job_id = str(uuid.uuid4())
        self.job_dir = Path(args.output_dir) / args.job_name / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        
        self.status = JobStatus.CREATED
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.generated_files = []
        self.error_message = None
        
        self.downloader = Downloader()
        self.generator = Generator.create_generator(args.model_type, args)
        self.cleanup = Cleanup(args) if (args.cleanup_local or args.cleanup_cache) else None
        
        logger.info(f"Job {self.job_id} created")
        self._save_status()
    
    def run_job(self, prompts: List[str]) -> bool:
        try:
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
            
        except Exception as e:
            self.error_message = str(e)
            self._update_status(JobStatus.FAILED)
            logger.error(f"Job {self.job_id} failed: {e}", exc_info=True)
            return False
    
    def _download_models(self) -> Dict[str, str]:
        logger.info("Downloading models and assets...")
        return self.downloader.download_all(self.args)
    
    def _generate_images(self, prompts: List[str], model_paths: Dict[str, str]) -> List[str]:
        logger.info(f"Generating images for {len(prompts)} prompts...")
        return self.generator.generate_batch(prompts, model_paths, self.job_dir)
    
    def _cleanup_files(self):
        logger.info("Cleaning up files...")
        self.cleanup.cleanup_job(self.generated_files, self.job_dir)
    
    def _update_status(self, status: JobStatus):
        self.status = status
        self.updated_at = datetime.now()
        self._save_status()
        logger.info(f"Job {self.job_id} status: {status.value}")
    
    def _save_status(self):
        status_data = {
            'job_id': self.job_id,
            'job_name': self.args.job_name,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'generated_files': self.generated_files,
            'error_message': self.error_message
        }
        
        status_file = self.job_dir / "job_status.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

