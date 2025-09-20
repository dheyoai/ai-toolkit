
import shutil
from pathlib import Path
from typing import List

from utils.logger import setup_logger

logger = setup_logger(__name__)


class Cleanup:
    
    def __init__(self, args):
        self.cleanup_local = args.cleanup_local
        self.cleanup_cache = args.cleanup_cache
    
    def cleanup_job(self, generated_files: List[str], job_dir: Path):
        if self.cleanup_local:
            self._cleanup_files(generated_files)
            self._cleanup_directory(job_dir)
        
        if self.cleanup_cache:
            self._cleanup_cache_dir()
    
    def _cleanup_files(self, file_paths: List[str]):
        for file_path in file_paths:
            try:
                Path(file_path).unlink()
                logger.info(f"Deleted: {Path(file_path).name}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
    
    def _cleanup_directory(self, dir_path: Path):
        try:
            shutil.rmtree(dir_path)
            logger.info(f"Deleted directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Failed to delete directory: {e}")
    
    def _cleanup_cache_dir(self):
        cache_dir = Path("./cache")
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info("Cleaned up model cache")
            except Exception as e:
                logger.warning(f"Failed to cleanup cache: {e}")

