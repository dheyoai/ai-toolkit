import shutil
from pathlib import Path
from typing import List, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CleanupError(Exception):
    pass


class Cleanup:
    def __init__(self):
        pass
    
    def cleanup_job(self, job_request: Dict[str, Any], generated_files: List[str]):
        try:
            self._validate_request(job_request)
            
            cleanup_local = job_request.get("cleanup_local", False)
            cleanup_cache = job_request.get("cleanup_cache", False)
            
            if not cleanup_local and not cleanup_cache:
                logger.info("No cleanup requested")
                return
            
            errors = []
            
            if cleanup_local:
                try:
                    self._cleanup_files(generated_files)
                    self._cleanup_output_directory(job_request)
                except Exception as e:
                    errors.append(f"Local cleanup failed: {str(e)}")
            
            if cleanup_cache:
                try:
                    self._cleanup_cache_dir(job_request)
                except Exception as e:
                    errors.append(f"Cache cleanup failed: {str(e)}")
            
            if errors:
                error_msg = "; ".join(errors)
                logger.warning(f"Cleanup completed with errors: {error_msg}")
            else:
                logger.info("Cleanup completed successfully")
                
        except Exception as e:
            logger.warning(f"Cleanup failed (non-critical): {str(e)}")
    
    def _validate_request(self, job_request: Dict[str, Any]):
        if not isinstance(job_request, dict):
            raise CleanupError("Job request must be a dictionary")
    
    def _cleanup_files(self, file_paths: List[str]):
        if not file_paths:
            logger.info("No files to cleanup")
            return
            
        failed_files = []
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted file: {path.name}")
                else:
                    logger.warning(f"File not found for deletion: {file_path}")
            except Exception as e:
                failed_files.append(f"{file_path}: {str(e)}")
        
        if failed_files:
            raise CleanupError(f"Failed to delete files: {'; '.join(failed_files[:3])}")
    
    def _cleanup_output_directory(self, job_request: Dict[str, Any]):
        try:
            output_dir = job_request.get("output_dir")
            job_name = job_request.get("job_name")
            
            if not output_dir or not job_name:
                logger.warning("Missing output_dir or job_name for directory cleanup")
                return
            
            job_base_dir = Path(output_dir) / job_name
            
            if job_base_dir.exists():
                shutil.rmtree(job_base_dir)
                logger.info(f"Deleted job directory: {job_base_dir}")
            else:
                logger.warning(f"Job directory not found for deletion: {job_base_dir}")
                
        except Exception as e:
            raise CleanupError(f"Failed to delete job directory: {str(e)}")
    
    def _cleanup_cache_dir(self, job_request: Dict[str, Any]):
        try:
            cache_dir = Path(job_request.get("cache_dir", "./cache"))
            
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"Cleaned up cache directory: {cache_dir}")
            else:
                logger.info("No cache directory found to clean")
                
        except Exception as e:
            raise CleanupError(f"Failed to cleanup cache: {str(e)}")
