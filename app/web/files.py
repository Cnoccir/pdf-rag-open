import json
import os
import boto3
import uuid
from botocore.config import Config
from botocore.exceptions import ClientError
import time
import tempfile
from typing import Tuple, Dict, Any, Optional, List
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def get_s3_client():
    try:
        config = Config(
            retries={'max_attempts': 3, 'mode': 'standard'},
            region_name=current_app.config['AWS_REGION']
        )

        return boto3.client('s3',
            aws_access_key_id=current_app.config['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=current_app.config['AWS_SECRET_ACCESS_KEY'],
            config=config
        )
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        raise

def generate_s3_key(pdf_id: str, file_type: str = 'pdf', page_num: Optional[int] = None, extra_id: Optional[str] = None) -> str:
    if file_type == 'pdf':
        return f"{pdf_id}.pdf"
    else:
        return f"{pdf_id}/{file_type}.json"

def get_s3_key(pdf_id: str) -> str:
    return f"{pdf_id}.pdf"

def upload(local_file_path: str, pdf_id: str = None) -> Tuple[Dict[str, str], int]:
    s3_client = get_s3_client()
    file_name = os.path.basename(local_file_path)
    if pdf_id is None:
        pdf_id = str(uuid.uuid4())
    s3_key = get_s3_key(pdf_id)
    try:
        s3_client.upload_file(local_file_path, current_app.config['AWS_BUCKET_NAME'], s3_key)
        logger.info(f"Successfully uploaded file {file_name} to S3 with key: {s3_key}")
        return {"file": pdf_id}, 200
    except ClientError as e:
        logger.error(f"Error uploading file {file_name}: {str(e)}")
        return {"error": str(e)}, 500

def create_download_url(file_id):
    return f"{current_app.config['UPLOAD_URL']}/{file_id}"

def download(file_id):
    return _Download(file_id)

class _Download:
    def __init__(self, file_id):
        self.file_id = file_id
        self.temp_dir = None
        self.file_path = ""
        self.s3_client = None

    def download(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, self.file_id)
        self.s3_client = boto3.client('s3',
            aws_access_key_id=current_app.config['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=current_app.config['AWS_SECRET_ACCESS_KEY'],
            region_name=current_app.config['AWS_REGION']
        )
        try:
            self.s3_client.download_file(current_app.config['AWS_BUCKET_NAME'], self.file_id, self.file_path)
        except ClientError as e:
            logger.error(f"Error downloading file {self.file_id}: {str(e)}")
            self.cleanup()
            return None
        return self.file_path

    def cleanup(self):
        if self.s3_client:
            self.s3_client.close()
        if self.temp_dir:
            retry_count = 3
            while retry_count > 0:
                try:
                    self.temp_dir.cleanup()
                    break
                except PermissionError:
                    logger.warning(f"PermissionError while cleaning up. Retrying in 1 second. Attempts left: {retry_count}")
                    time.sleep(1)
                    retry_count -= 1
            if retry_count == 0:
                logger.error(f"Failed to clean up temporary directory for file {self.file_id}")

    def __enter__(self):
        return self.download()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

# New function to delete a file from S3
def delete_file(file_id):
    s3_client = get_s3_client()
    try:
        s3_client.delete_object(Bucket=current_app.config['AWS_BUCKET_NAME'], Key=file_id)
        logger.info(f"File {file_id} deleted successfully from S3.")
        return True
    except ClientError as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        return False
"""
def upload_content(content: str, s3_key: str) -> Tuple[Dict[str, str], int]:
    s3_client = get_s3_client()
    try:
        s3_client.put_object(Body=content, Bucket=current_app.config['AWS_BUCKET_NAME'], Key=s3_key)
        logger.info(f"Successfully uploaded content to S3 with key: {s3_key}")
        return {"file": s3_key}, 200
    except Exception as e:
        logger.error(f"Error uploading content: {str(e)}")
        return {"error": str(e)}, 500
"""
def download_file_content(pdf_id: str, file_type: str = 'pdf'):
    s3_client = get_s3_client()
    s3_key = generate_s3_key(pdf_id, file_type=file_type)

    try:
        response = s3_client.get_object(Bucket=current_app.config['AWS_BUCKET_NAME'], Key=s3_key)
        logger.info(f"Successfully downloaded content for file: {pdf_id} with S3 key: {s3_key}")
        return response['Body'].read()
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f"File not found in S3: {s3_key}")
        else:
            logger.error(f"Error downloading file content for {pdf_id} with S3 key {s3_key}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading file content for {pdf_id} with S3 key {s3_key}: {str(e)}")
        return None

def ensure_file_in_s3(pdf_id: str) -> bool:
    s3_client = get_s3_client()
    s3_key = get_s3_key(pdf_id)
    try:
        s3_client.head_object(Bucket=current_app.config['AWS_BUCKET_NAME'], Key=s3_key)
        logger.info(f"File {pdf_id} exists in S3 with key: {s3_key}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.warning(f"File {pdf_id} not found in S3 with key: {s3_key}")
            return False
        else:
            logger.error(f"Error checking file {pdf_id} in S3 with key {s3_key}: {str(e)}")
            raise
