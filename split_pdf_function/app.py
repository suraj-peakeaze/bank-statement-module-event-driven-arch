import collections.abc
from PyPDF2 import PdfReader
from botocore import retries
from utils.dynamo_utils import update_record_table, get_items_from_record_table

collections.Sequence = collections.abc.Sequence

import boto3
import json
import tempfile
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from botocore.client import ClientError
from urllib.parse import unquote_plus
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv()

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append("/opt/python")

from utils.split_utils import split_pdf_into_pages
from utils.s3_utils import download_file, upload_file, get_job_id_from_s3_metadata

# AWS clients - Initialize once outside handler for better performance
dynamodb = boto3.resource("dynamodb")
sqs = boto3.client("sqs")

# Environment variables
BUCKET = os.getenv("AWS_STORAGE_BUCKET_NAME")
RECORD_TABLE = os.getenv("JOB_STATUS_TABLE")
PROCESS_PAGE_QUEUE_URL = os.getenv("PROCESS_PAGE_QUEUE_URL")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


def extract_event_data(event):
    """Extract job_id and pdf_key from different event types"""
    job_id = None
    pdf_key = None
    bucket = BUCKET

    if "Records" in event:
        record = event["Records"][0]

        if "s3" in record:
            # S3 event
            bucket = record["s3"]["bucket"]["name"]
            pdf_key = unquote_plus(record["s3"]["object"]["key"])
            logger.info(f"Processing S3 object: s3://{bucket}/{pdf_key}")

            # Find the corresponding DynamoDB record
            job_id = get_job_id_from_s3_metadata(bucket, pdf_key)
            if not job_id:
                raise ValueError(f"No DynamoDB record found for S3 key: {pdf_key}")

        elif "body" in record:
            # SQS event
            message_body = json.loads(record["body"])
            job_id = message_body.get("job_id")
            pdf_key = message_body.get("pdf_key")
        else:
            raise ValueError(f"Unknown event type in Records: {record.keys()}")
    else:
        # Direct invocation
        job_id = event.get("job_id")
        pdf_key = event.get("pdf_key")

    if not job_id:
        raise ValueError("job_id not found in event")

    return job_id, pdf_key, bucket


def send_sqs_message(
    job_id, page_number, bucket, user_email, pdf_name, page_s3_key, length_of_pdf, table
):
    """Send SQS message for page processing"""
    sqs_message = {
        "job_id": job_id,
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
            current = get_items_from_record_table(job_id, page_number)
            retries = current.get("retries")
            if current and current[0].get("status") != "COMPLETED":
                status = "UN_PROCESSABLE" if retries >= 3 else "FAILED"
                update_record_table(
                    job_id, page_number, {"status": status, "retries": retries}
                )  # Send SQS message
                message_sent = send_sqs_message(
                    job_id,
                    page_number,
                    bucket,
                    user_email,
                    pdf_name,
                    page_s3_key,
                    length_of_pdf,
                    table,
                )
        except Exception as db_error:
            logger.error(f"Error updating page status to FAILED: {db_error}")

        return False


def update_job_status_on_error(job_id, page_number, error, retries):
    """Update job status when an error occurs"""
    status = "UN_PROCESSABLE" if retries >= MAX_RETRIES else "FAILED"

    error_record = {
        "retries": retries,
        "status": status,
        "error": str(error),
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        update_record_table(
            job_id=job_id, page_number=page_number, update_data=error_record
        )
        logger.info(f"Updated job {job_id} with error status: {status}")
    except ClientError as db_error:
        logger.error(f"DynamoDB update error for job {job_id}: {db_error}")


def lambda_handler(event, context):
    """Lambda function to split PDF into individual pages"""

    # Initialize variables for error handling
    job_id = None
    retries = 0
    table = dynamodb.Table(RECORD_TABLE)
    page_number = 0

    try:
        logger.info(f"Received event: {json.dumps(event, default=str)}")

        # Extract event data
        job_id, pdf_key, bucket = extract_event_data(event)

        # Get item from record table
        items = get_items_from_record_table(job_id)
        item = items[0]
        if not item:
            raise ValueError(f"No record found for job_id: {job_id}")

        logger.info(item)

        # Calculate retries
        current_retries = int(item.get("retries", 0))
        retries = current_retries + 1

        # Extract data from item
        if not pdf_key:
            pdf_key = item.get("pdf_key")
        user_email = item.get("user_email")

        if not pdf_key:
            raise ValueError("pdf_key not found in event or DynamoDB record")

        pdf_name = pdf_key.split("/")[-1]

        # Download PDF from S3
        local_pdf = "/tmp/input.pdf"
        download_file(bucket, pdf_key, local_pdf)

        # Read PDF and get page count
        reader = PdfReader(local_pdf)
        length_of_pdf = len(reader.pages)
        logger.info(f"Processing PDF with {length_of_pdf} pages")

        # Validate PDF has pages
        if length_of_pdf == 0:
            raise ValueError("PDF has no pages to process")

        # Split PDF into pages
        with tempfile.TemporaryDirectory() as temp_dir:
            page_paths = split_pdf_into_pages(local_pdf, output_dir=temp_dir)

            if len(page_paths) != length_of_pdf:
                logger.warning(
                    f"Expected {length_of_pdf} pages, got {len(page_paths)} files"
                )

            # Process each page
            page_keys = []
            successful_pages = 0
            failed_pages = 0

            for i, page_path in enumerate(page_paths):
                page_number = i + 1
                page_s3_key = f"split_pages/{job_id}/page_{page_number}.pdf"

                try:
                    # Upload page to S3
                    upload_file(page_path, bucket, page_s3_key)
                    page_keys.append(page_s3_key)

                    # Send SQS message
                    message_sent = send_sqs_message(
                        job_id,
                        page_number,
                        bucket,
                        user_email,
                        pdf_name,
                        page_s3_key,
                        length_of_pdf,
                        table,
                    )

                    # Update main job record with processing status
                    final_status = (
                        "PROCESSING_PAGES" if successful_pages > 0 else "FAILED"
                    )
                    record_updated = update_record_table(
                        job_id=job_id,
                        page_number=page_number,
                        update_data={
                            "status": final_status,
                            "job_id": job_id,
                            "pdf_key": pdf_key,
                            "page_s3_key": page_s3_key,
                            "length_of_pdf": length_of_pdf,
                            "user_email": user_email,
                            "total_pages": length_of_pdf,
                            "successful_pages": successful_pages,
                            "failed_pages": failed_pages,
                            "updated_at": datetime.utcnow().isoformat(),
                            "retries": current_retries,
                            "next": "ProcessPages",
                            "calling_gemini_for_header_extraction": False,
                            "gemini_header_extraction_completed": False,
                            "calling_gemini_cleaning_process": False,
                            "gemini_cleaning_process_completed": False,
                        },
                    )

                    if record_updated and message_sent:
                        successful_pages += 1
                    else:
                        failed_pages += 1

                except Exception as page_error:
                    logger.error(f"Error processing page {page_number}: {page_error}")
                    failed_pages += 1
                    continue

            final_status = (
                "PROCESSING_PAGES" if successful_pages == length_of_pdf else "FAILED"
            )
            update_record_table(
                job_id=job_id,
                page_number=0,
                update_data={
                    "status": final_status,
                    "next": "ProcessPages",
                    "length_of_pdf": length_of_pdf,
                },
            )

        # Clean up local file
        try:
            os.remove(local_pdf)
        except:
            pass  # Ignore cleanup errors

        success_message = (
            f"Successfully processed {successful_pages}/{length_of_pdf} pages"
        )
        if failed_pages > 0:
            success_message += f" ({failed_pages} failed)"

        logger.info(success_message)

        return {
            "statusCode": 200,
            "job_id": job_id,
            "pages_created": successful_pages,
            "pages_failed": failed_pages,
            "total_pages": length_of_pdf,
            "message": success_message,
        }

    except Exception as e:
        logger.error(f"Error in split_pdf_handler: {str(e)}")

        if job_id:
            update_job_status_on_error(job_id, page_number, e, retries)

        # Re-raise the exception for Lambda error handling
        raise e
