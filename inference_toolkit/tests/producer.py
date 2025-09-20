#!/usr/bin/env python3
"""
Producer: Create request -> Push to queue -> Wait for response -> Print result
"""

import json
import time
import redis


def create_request():
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


def push_request():
    redis_client = redis.Redis(decode_responses=True)
    request = create_request()
    request_json = json.dumps(request)
    
    redis_client.rpush("inference_requests", request_json)
    print(f"Pushed request: {request['job_name']}")


def wait_for_response():
    redis_client = redis.Redis(decode_responses=True)
    
    print("Waiting for response...")
    while True:
        result = redis_client.blpop("inference_responses", timeout=300)
        if result:
            response_data = result[1]
            response = json.loads(response_data)
            print_response(response)
            break
        else:
            print("Timeout waiting for response")
            break


def print_response(response):
    print("\nJob Response:")
    print(f"Job ID: {response.get('job_id')}")
    print(f"Job Name: {response.get('job_name')}")
    print(f"Status: {response.get('status')}")
    print(f"Success: {response.get('success')}")
    
    uploaded_files = response.get('uploaded_files', [])
    if uploaded_files:
        print(f"Uploaded Files ({len(uploaded_files)}):")
        for url in uploaded_files:
            print(f"  {url}")
    
    if response.get('error_message'):
        print(f"Error: {response.get('error_message')}")


def clear_queues():
    redis_client = redis.Redis(decode_responses=True)
    redis_client.delete("inference_requests")
    redis_client.delete("inference_responses")
    print("Cleared queues")


def main():
    clear_queues()
    push_request()
    wait_for_response()


if __name__ == "__main__":
    main()