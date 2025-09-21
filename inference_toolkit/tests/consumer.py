
import json
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.job_orchestrator import JobOrchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Consumer for processing jobs from Redis queue")
    
    parser.add_argument("--request_queue", type=str, 
                       default=os.getenv("REDIS_REQUEST_QUEUE", "inference_requests"),
                       help="Redis queue name for job requests")
    parser.add_argument("--response_queue", type=str, 
                       default=os.getenv("REDIS_RESPONSE_QUEUE", "inference_responses"),
                       help="Redis queue name for job responses")
    parser.add_argument("--timeout", type=int, 
                       default=int(os.getenv("REDIS_TIMEOUT", "30")),
                       help="Timeout in seconds for waiting for jobs")
    parser.add_argument("--uploader_type", type=str, 
                       default=os.getenv("UPLOADER_TYPE", "s3"),
                       choices=["s3"], help="Type of uploader to use")
    
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        
        logger.info(f"Starting consumer with queues: {args.request_queue} -> {args.response_queue}")
        logger.info(f"Timeout: {args.timeout}s, Uploader: {args.uploader_type}")
        
        orchestrator = JobOrchestrator(
            request_queue=args.request_queue,
            response_queue=args.response_queue,
            timeout=args.timeout,
            uploader_type=args.uploader_type
        )
        
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"Consumer failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()