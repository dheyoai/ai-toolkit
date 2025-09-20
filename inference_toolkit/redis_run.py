import json
import time
import redis
from services.redis_processor import RedisProcessor
from core.job_orchestrator import JobOrchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_test_request():
    return {
        "job_name": "allu_arjun_alia_test",
        "model_type": "qwen",
        "model_path": "Qwen/Qwen-Image",
        "hf_lora_id": "DheyoAI/allu_arjun_and_alia_bhatt_1",
        "instruction": "A close-up portrait of two actors sitting together in a cozy cafe, warm ambient lighting, soft bokeh background, both facing the camera with gentle smiles",
        "num_inference_steps": 50,
        "aspect_ratio": "16:9",
        "num_images_per_prompt": 2,
        "true_cfg_scale": 4.0,
        "seed": 42,
        "dtype": "bf16",
        "output_dir": "./outputs"
    }


def main():
    print("Starting Redis processor test...")
    
    request_queue = "inference_requests"
    response_queue = "inference_responses"
    
    try:
        # 1. Initialize Redis processor
        processor = RedisProcessor()
        print("Redis processor connected")
        
        # 2. Clear queues
        processor.clear_queue(request_queue)
        processor.clear_queue(response_queue)
        print("Queues cleared")
        
        # 3. Push test request
        redis_client = redis.Redis(decode_responses=True)
        request = create_test_request()
        request_json = json.dumps(request, indent=2)
        redis_client.rpush(request_queue, request_json)
        print("Test request pushed to queue")
        
        # 4. Pop request and convert to args
        args = processor.pop_request(request_queue, timeout=5)
        if args is None:
            print("No request received")
            return
        
        print("Request popped and converted to args")
        print(f"  Job name: {args.job_name}")
        print(f"  Model: {args.model_path}")
        print(f"  LoRA: {args.hf_lora_id}")
        
        # 5. Run the actual job
        print("\n Running job...")
        prompts = [args.instruction]
        
        orchestrator = JobOrchestrator(args)
        success = orchestrator.run_job(prompts)
        
        # 6. Get job result
        job_result = orchestrator.get_status_dict()
        job_result['success'] = success
        
        if success:
            print("Job completed successfully")
            print(f"  Generated {len(job_result.get('generated_files', []))} files")
        else:
            print("Job failed")
            print(f"  Error: {job_result.get('error_message', 'Unknown error')}")
        
        # 7. Push response
        processor.push_response(response_queue, job_result)
        print("Response pushed to queue")
        
        # 8. Check response
        response_data = redis_client.lpop(response_queue)
        if response_data:
            response = json.loads(response_data)
            print("\n Response received:")
            print(f"  Job ID: {response.get('job_id')}")
            print(f"  Status: {response.get('status')}")
            print(f"  Success: {response.get('success')}")
            print(f"  Files: {len(response.get('generated_files', []))}")
        
        print("\n Test completed successfully!")
        
    except Exception as e:
        print(f"\n Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()