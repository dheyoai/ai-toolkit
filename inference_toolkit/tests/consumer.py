#!/usr/bin/env python3
"""
Worker: Wait for request -> Process job -> Upload to S3 -> Push response
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.redis_processor import RedisProcessor
from services.uploader import Uploader
from core.job_orchestrator import JobOrchestrator


def process_job():
    redis_processor = RedisProcessor()
    uploader = Uploader.create_uploader("s3")
    
    print("Waiting for job request...")
    args = redis_processor.pop_request("inference_requests", timeout=120)
    
    if args is None:
        print("No job received")
        return
    
    print(f"Processing job: {args.job_name}")
    
    orchestrator = JobOrchestrator(args)
    success = orchestrator.run_job([args.instruction])
    
    job_result = orchestrator.get_status_dict()
    job_result['success'] = success
    
    if success:
        generated_files = job_result.get('generated_files', [])
        print(f"Job completed. Generated {len(generated_files)} files")
        
        if generated_files:
            user_id = f"user_{args.job_name}"
            generation_id = job_result['job_id']
            
            upload_results = uploader.upload_generated_files(
                generated_files=generated_files,
                user_id=user_id,
                generation_id=generation_id
            )
            
            successful_uploads = [r for r in upload_results if r['success']]
            print(f"Uploaded {len(successful_uploads)}/{len(upload_results)} files")
            
            job_result['uploaded_files'] = [r['s3_url'] for r in successful_uploads]
        else:
            job_result['uploaded_files'] = []
    else:
        print("Job failed")
        job_result['uploaded_files'] = []
    
    redis_processor.push_response("inference_responses", job_result)
    print("Response pushed to queue")


def main():
    try:
        process_job()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()