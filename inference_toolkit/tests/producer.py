import json
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from services.redis_processor import RedisProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_request():
    return {
        "generation_id": "02b0db75-77f6-45e1-85c4-0ca2b1185ff3",
        "user_id": "aakashvarma",
        "instruction": "A close-up portrait of (([A] man)) and (([AB] woman)) sitting together in a cozy cafe, warm ambient lighting, soft bokeh background, both facing the camera with gentle smiles",
        "hf_lora_id": "DheyoAI/allu_arjun_and_alia_bhatt_1"
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Producer for sending jobs to Redis queue")
    
    parser.add_argument("--request_queue", type=str, 
                       default=os.getenv("REDIS_REQUEST_QUEUE", "inference_requests"),
                       help="Redis queue name for job requests")
    parser.add_argument("--response_queue", type=str, 
                       default=os.getenv("REDIS_RESPONSE_QUEUE", "inference_responses"),
                       help="Redis queue name for job responses")
    parser.add_argument("--timeout", type=int, 
                       default=int(os.getenv("REDIS_TIMEOUT", "300")),
                       help="Timeout in seconds for waiting for response")
    parser.add_argument("--clear_queues", action="store_true", 
                       help="Clear queues before starting")
    parser.add_argument("--no_wait", action="store_true", 
                       help="Don't wait for response after sending request")
    
    return parser.parse_args()


def push_request(redis_processor, request_queue, request):
    try:
        request_json = json.dumps(request)
        redis_processor.redis_client.rpush(request_queue, request_json)
        logger.info(f"Pushed request to {request_queue}: {request['generation_id']}")
        return True
    except Exception as e:
        logger.error(f"Failed to push request: {str(e)}")
        return False


def wait_for_response(redis_processor, response_queue, timeout):
    logger.info(f"Waiting for response from {response_queue}...")
    
    try:
        while True:
            result = redis_processor.redis_client.blpop(response_queue, timeout=timeout)
            if result:
                response_data = result[1]
                response = json.loads(response_data)
                print(response)
                return True
            else:
                logger.warning(f"Timeout waiting for response ({timeout}s)")
                return False
    except Exception as e:
        logger.error(f"Error waiting for response: {str(e)}")
        return False



def clear_queues(redis_processor, request_queue, response_queue):
    try:
        success1 = redis_processor.clear_queue(request_queue)
        success2 = redis_processor.clear_queue(response_queue)
        
        if success1 and success2:
            logger.info(f"Successfully cleared queues: {request_queue}, {response_queue}")
        else:
            logger.warning(f"Failed to clear some queues")
        
        return success1 and success2
    except Exception as e:
        logger.error(f"Error clearing queues: {str(e)}")
        return False


def validate_request(request):
    required_fields = [
        "generation_id", "user_id", "instruction", "hf_lora_id"
    ]
    
    missing_fields = [field for field in required_fields if field not in request]
    if missing_fields:
        logger.error(f"Request missing required fields: {missing_fields}")
        return False
    
    logger.info("Request validation passed")
    return True


def main():
    try:
        args = parse_args()
        
        logger.info("Initializing Redis processor...")
        redis_processor = RedisProcessor()
        
        logger.info(f"Using queues: {args.request_queue} -> {args.response_queue}")
        
        if args.clear_queues:
            logger.info("Clearing queues...")
            if not clear_queues(redis_processor, args.request_queue, args.response_queue):
                logger.error("Failed to clear queues")
                return
        
        logger.info("Creating job request...")
        request = create_request()
        
        if not validate_request(request):
            logger.error("Request validation failed")
            return
        
        logger.info("Pushing job request...")
        if not push_request(redis_processor, args.request_queue, request):
            logger.error("Failed to push request")
            return
        
        if not args.no_wait:
            if not wait_for_response(redis_processor, args.response_queue, args.timeout):
                logger.warning("Did not receive response within timeout")
        else:
            logger.info("Request sent successfully (not waiting for response)")
        
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    except Exception as e:
        logger.error(f"Producer failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()