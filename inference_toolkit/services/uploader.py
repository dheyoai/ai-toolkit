import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from utils.logger import setup_logger

logger = setup_logger(__name__)


class UploaderError(Exception):
    pass


class S3Uploader:
    def __init__(self):
        self._validate_environment()
        self._initialize_client()
    
    def _validate_environment(self):
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = os.getenv('AWS_DEFAULT_REGION', 'us-west-1')
        
        required_vars = [
            ('S3_BUCKET_NAME', self.bucket_name),
            ('AWS_ACCESS_KEY_ID', self.aws_access_key_id),
            ('AWS_SECRET_ACCESS_KEY', self.aws_secret_access_key)
        ]
        
        missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
        if missing_vars:
            raise UploaderError(f"Missing required environment variables: {missing_vars}")
    
    def _initialize_client(self):
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
            
            self._test_connection()
            
            logger.info(f"S3 uploader initialized for bucket: {self.bucket_name} in region: {self.region_name}")
            
        except Exception as e:
            raise UploaderError(f"Failed to initialize S3 client: {str(e)}")
    
    def upload_generated_files(self, generated_files: List[str], 
                             user_id: str, 
                             generation_id: Optional[str] = None) -> List[Dict[str, str]]:
        try:
            self._validate_upload_inputs(generated_files, user_id)
            
            if not generation_id:
                generation_id = str(uuid.uuid4())
            
            logger.info(f"Starting upload for {len(generated_files)} files")
            logger.info(f"User ID: {user_id}, Generation ID: {generation_id}")
            
            upload_results = []
            successful_uploads = 0
            
            for file_path in generated_files:
                try:
                    result = self._upload_single_file(file_path, user_id, generation_id)
                    upload_results.append(result)
                    
                    if result['success']:
                        successful_uploads += 1
                        logger.info(f"Uploaded: {result['s3_url']}")
                    else:
                        logger.error(f"Failed: {file_path} - {result['error']}")
                        
                except Exception as e:
                    error_msg = f"Unexpected error uploading {file_path}: {str(e)}"
                    logger.error(error_msg)
                    upload_results.append({
                        'local_path': file_path,
                        's3_url': None,
                        's3_key': None,
                        'success': False,
                        'error': error_msg
                    })
            
            logger.info(f"Upload completed: {successful_uploads}/{len(generated_files)} files successful")
            
            if successful_uploads == 0:
                raise UploaderError("All file uploads failed")
            
            return upload_results
            
        except UploaderError:
            raise
        except Exception as e:
            raise UploaderError(f"Upload process failed: {str(e)}")
    
    def _validate_upload_inputs(self, generated_files: List[str], user_id: str):
        if not generated_files:
            raise UploaderError("No files provided for upload")
        
        if not user_id or not user_id.strip():
            raise UploaderError("User ID is required for upload")
        
        if not isinstance(generated_files, list):
            raise UploaderError("Generated files must be a list")
    
    def _upload_single_file(self, file_path: str, user_id: str, generation_id: str) -> Dict[str, str]:
        try:
            local_path = Path(file_path)
            
            if not local_path.exists():
                return {
                    'local_path': file_path,
                    's3_url': None,
                    's3_key': None,
                    'success': False,
                    'error': f"File does not exist: {file_path}"
                }
            
            if local_path.stat().st_size == 0:
                return {
                    'local_path': file_path,
                    's3_url': None,
                    's3_key': None,
                    'success': False,
                    'error': f"File is empty: {file_path}"
                }

            s3_key = self._generate_s3_key(local_path, user_id, generation_id)

            extra_args = {
                'ContentType': self._get_content_type(local_path),
                'Metadata': {
                    'user-id': user_id,
                    'generation-id': generation_id,
                    'original-filename': local_path.name
                }
            }
            
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            s3_url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            
            return {
                'local_path': file_path,
                's3_url': s3_url,
                's3_key': s3_key,
                'success': True,
                'error': None
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"AWS error ({error_code}): {str(e)}"
            return {
                'local_path': file_path,
                's3_url': None,
                's3_key': None,
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            return {
                'local_path': file_path,
                's3_url': None,
                's3_key': None,
                'success': False,
                'error': f"Upload failed: {str(e)}"
            }
    
    def _generate_s3_key(self, file_path: Path, user_id: str, generation_id: str) -> str:
        filename = file_path.name
        key_prefix = os.getenv('S3_KEY_PREFIX', 'images')
        s3_key = f"{key_prefix}/{user_id}/{generation_id}/{filename}"
        return s3_key
    
    def _get_content_type(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    def _test_connection(self):
        try:
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise UploaderError(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                raise UploaderError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise UploaderError(f"S3 connection test failed: {str(e)}")
        except NoCredentialsError:
            raise UploaderError("AWS credentials not found in environment variables")
        except Exception as e:
            raise UploaderError(f"S3 connection test failed: {str(e)}")


class Uploader:
    @staticmethod
    def create_uploader(upload_type: str = "s3") -> S3Uploader:
        try:
            supported_types = ["s3"]
            if upload_type not in supported_types:
                raise UploaderError(f"Unsupported uploader type: {upload_type}. Supported types: {supported_types}")
            
            if upload_type == "s3":
                return S3Uploader()
                
        except Exception as e:
            if isinstance(e, UploaderError):
                raise
            raise UploaderError(f"Failed to create uploader: {str(e)}")