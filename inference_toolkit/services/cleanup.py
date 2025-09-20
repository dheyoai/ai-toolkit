import shutil
from pathlib import Path
from typing import List

from utils.logger import setup_logger

logger = setup_logger(__name__)


class CleanupError(Exception):
    pass


class Cleanup:
    
    def __init__(self, args):
        self.cleanup_local = args.cleanup_local
        self.cleanup_cache = args.cleanup_cache
    
    def cleanup_job(self, generated_files: List[str], job_dir: Path):
        errors = []
        
        if self.cleanup_local:
            try:
                self._cleanup_files(generated_files)
            except Exception as e:
                errors.append(f"File cleanup failed: {str(e)}")
            
            try:
                self._cleanup_directory(job_dir)
            except Exception as e:
                errors.append(f"Directory cleanup failed: {str(e)}")
        
        if self.cleanup_cache:
            try:
                self._cleanup_cache_dir()
            except Exception as e:
                errors.append(f"Cache cleanup failed: {str(e)}")
        
        if errors:
            error_msg = "; ".join(errors)
            logger.warning(f"Cleanup completed with errors: {error_msg}")
            # Don't raise exception for cleanup errors
        else:
            logger.info("Cleanup completed successfully")
    
    def _cleanup_files(self, file_paths: List[str]):
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
    
    def _cleanup_directory(self, dir_path: Path):
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"Deleted directory: {dir_path}")
            else:
                logger.warning(f"Directory not found for deletion: {dir_path}")
        except Exception as e:
            raise CleanupError(f"Failed to delete directory {dir_path}: {str(e)}")
    
    def _cleanup_cache_dir(self):
        cache_dir = Path("./cache")
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info("Cleaned up model cache")
            else:
                logger.info("No cache directory found to clean")
        except Exception as e:
            raise CleanupError(f"Failed to cleanup cache: {str(e)}")