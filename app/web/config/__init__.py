import os
from dotenv import load_dotenv
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

load_dotenv()


class Config:
    SESSION_PERMANENT = True
    SECRET_KEY = os.environ["SECRET_KEY"]
    SQLALCHEMY_DATABASE_URI = os.environ["SQLALCHEMY_DATABASE_URI"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_BUCKET_NAME = "pdf-rag-bucket"
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
    UPLOAD_URL = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com"
    CELERY = {
        "broker_url": os.environ.get("REDIS_URI", False),
        "task_ignore_result": True,
        "broker_connection_retry_on_startup": False,
    }
