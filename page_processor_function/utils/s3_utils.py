import boto3
import os
from io import BytesIO, IOBase
import pandas as pd
from dotenv import load_dotenv
import logging
import json

load_dotenv()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sqs = boto3.client("sqs")

PROCESS_PAGE_QUEUE_URL = os.getenv("PROCESS_PAGE_QUEUE_URL")


# === Client Setup ===
def get_s3_client():
    """
    Returns a boto3 S3 client.
    Supports custom endpoint for local testing (LocalStack).
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    return (
        boto3.client("s3", endpoint_url=endpoint_url)
        if endpoint_url
        else boto3.client("s3")
    )


# === File Downloads ===
def download_file(bucket, key, local_path):
    """
    Download a file from S3 to a local file path.
    Creates local directory if needed.
    """
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket, key, local_path)
        print(f"✅ Downloaded s3://{bucket}/{key} to {local_path}")
    except Exception as e:
        print(f"❌ Failed to download s3://{bucket}/{key}: {e}")
        raise


def download_fileobj(bucket, key):
    """
    Download a file from S3 to memory (BytesIO).
    """
    s3 = get_s3_client()
    buf = BytesIO()

    try:
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        print(f"✅ Downloaded s3://{bucket}/{key} to memory")
        return buf
    except Exception as e:
        print(f"❌ Failed to download fileobj s3://{bucket}/{key}: {e}")
        raise


# === File Uploads ===
def upload_file(local_path, bucket, key):
    """
    Upload a file from local disk (/tmp only in Lambda) to S3.
    """
    s3 = get_s3_client()

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found for upload: {local_path}")

    try:
        s3.upload_file(local_path, bucket, key)
        print(f"✅ Uploaded {local_path} to s3://{bucket}/{key}")
    except Exception as e:
        print(f"❌ Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
        raise


def upload_fileobj(file_obj, bucket, key):
    """
    Upload an in-memory file-like object (BytesIO) to S3.
    """
    s3 = get_s3_client()

    if not isinstance(file_obj, IOBase):
        raise TypeError("file_obj must be a file-like object")

    file_obj.seek(0)
    try:
        s3.upload_fileobj(file_obj, bucket, key)
        print(f"✅ Uploaded file object to s3://{bucket}/{key}")
    except Exception as e:
        print(f"❌ Failed to upload file object to s3://{bucket}/{key}: {e}")
        raise


# === CSV Helpers ===
def read_csv_from_s3(bucket, key):
    """
    Read a CSV from S3 directly into a pandas DataFrame (in-memory).
    """
    buf = download_fileobj(bucket, key)
    return pd.read_csv(buf)


def write_csv_to_s3(df, bucket, key):
    """
    Write a pandas DataFrame to CSV and upload to S3 in-memory.
    """
    buf = BytesIO()
    df.to_csv(buf, index=False)
    upload_fileobj(buf, bucket, key)


def send_sqs_message(
    job_id, page_number, bucket, user_email, pdf_name, page_s3_key, length_of_pdf, table
):
    """Send SQS message for page processing"""
    sqs_message = {
        "job_id": job_id,
        "page_id": f"{job_id}_{page_number}",
        "bucket": bucket,
        "user_email": user_email,
        "pdf_name": pdf_name,
        "page_number": page_number,
        "page_key": page_s3_key,
        "total_pages": length_of_pdf,
    }

    try:
        sqs.send_message(
            QueueUrl=PROCESS_PAGE_QUEUE_URL,
            MessageBody=json.dumps(sqs_message),
        )
        logger.info(f"Sent SQS message for page {page_number}")
        return True
    except Exception as e:
        logger.error(f"Error sending SQS message for page {page_number}: {e}")

        # Update page status to failed
        try:
            table.update_item(
                Key={"page_id": f"{job_id}_{page_number}"},
                UpdateExpression="SET #status = :status, #error = :error",
                ExpressionAttributeNames={
                    "#status": "status",
                    "#error": "error",
                },
                ExpressionAttributeValues={
                    ":status": "FAILED",
                    ":error": str(e),
                },
            )
        except Exception as db_error:
            logger.error(f"Error updating page status to FAILED: {db_error}")

        return False
