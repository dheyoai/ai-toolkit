#!/bin/bash

echo "Setting up environment variables for Image Generation Service..."

# Redis Configuration
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"
export REDIS_PASSWORD=""  # Leave empty if no password

# Redis Queue Names
export REDIS_REQUEST_QUEUE="your-request-queue"
export REDIS_RESPONSE_QUEUE="your-resposne-queue"
export REDIS_TIMEOUT="5"

# AWS S3 Configuration
export UPLOADER_TYPE="s3"
export AWS_ACCESS_KEY_ID="your_aws_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key_here"
export AWS_DEFAULT_REGION="your-aws-region"
export S3_BUCKET_NAME="your-s3-bucket-name"
export S3_KEY_PREFIX="images"

# Default Directories
export DEFAULT_OUTPUT_DIR="/dheyo/lora-infer/outputs"
export DEFAULT_CACHE_DIR="/dheyo/lora-infer/cache"

