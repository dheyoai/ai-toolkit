import json
import time
import redis

def create_request():
    return {
        "job_name": "allu_arjun_alia_test",
        "model_type": "qwen",
        "model_path": "Qwen/Qwen-Image",
        "cache_dir": "/dheyo/lora-infer",
        "hf_lora_id": "DheyoAI/allu_arjun_and_alia_bhatt_1",
        "instruction": "A close-up portrait of (([A] man)) and (([AB] woman)) sitting together in a cozy cafe, warm ambient lighting, soft bokeh background, both facing the camera with gentle smiles",
        "num_images_per_prompt": 2,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "aspect_ratio": "16:9",
        "seed": 42,
        "dtype": "bf16",
        "negative_prompt": "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark",
        "output_dir": "./outputs",
        "cleanup_local": False,
        "cleanup_cache": False,
        "local_lora_config": False
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