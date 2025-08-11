import json
import pandas as pd
import tempfile
import os
from datetime import datetime
import boto3
import logging

from utils.s3_utils import download_file, upload_file
from utils.email_service import EmailService
from utils.dynamo_utils import (
    get_items_from_record_0_table,
    get_items_from_record_table,
    update_record_table,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")


def lambda_handler(event, context):
    """Aggregate processed pages for a job into final CSV."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure job_id is always string
            job_id = int(event["detail"]["job_id"])
            page_number = event["detail"]["page_number"] or 0
            bucket = event["detail"]["bucket"]

            if not bucket:
                raise ValueError("No S3 bucket specified")

            # Get metadata for page 0
            items = get_items_from_record_0_table(
                job_id=job_id, page_number=page_number
            )
            if not items:
                raise ValueError(f"No record found for job_id {job_id}")

            item = items[0]
            user_email = item.get("user_email", "")
            length_of_pdf = int(item.get("length_of_pdf"))
            pdf_key = item.get("pdf_key")
            pdf_name = pdf_key.split("/")[-1]
            header = item.get("header", [])

            # Collect all page info
            pages_metadata = []
            for page_num in range(1, length_of_pdf + 1):
                page_items = get_items_from_record_table(
                    job_id=job_id, page_number=page_num
                )
                if not page_items:
                    logger.warning(f"No record for page {page_num} in job {job_id}")
                    continue
                pages_metadata.append(page_items[0])

            # Sort pages just to be safe
            pages_metadata.sort(key=lambda x: x.get("page_number", 0))

            # Merge CSVs
            final_df = merge_csvs(pages_metadata, bucket, header, temp_dir)
            if final_df is None or final_df.empty:
                return {
                    "job_id": job_id,
                    "status": "FAILED",
                    "error": "No valid CSV data",
                }

            # Save final CSV to S3
            final_s3_key = f"final_outputs/{job_id}/{pdf_name}_final.csv"
            save_csv_to_s3(final_df, bucket, final_s3_key)

            # Generate presigned URL
            download_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": final_s3_key},
                ExpiresIn=3600,
            )

            # Send notification
            send_notification(
                user_email,
                pdf_name,
                len(pages_metadata),
                download_url,
                job_id,
                length_of_pdf,
            )

            logger.info(f"Job {job_id} completed successfully")
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "final_csv_key": final_s3_key,
                "download_url": download_url,
                "record_count": len(final_df),
            }

    except Exception as e:
        logger.exception(f"Error processing job: {e}")
        return {
            "job_id": event.get("job_id", "UNKNOWN"),
            "status": "FAILED",
            "error": str(e),
        }


def merge_csvs(pages, bucket, header, temp_dir):
    """Download and merge CSV files from S3 keys."""
    dfs = []

    for page in pages:
        csv_key = page.get("cleaned_csv_key")
        page_num = page.get("page_number")

        if not csv_key:
            logger.info(f"Skipping page {page_num} - no CSV key")
            continue

        csv_path = os.path.join(temp_dir, f"page_{page_num}.csv")
        try:
            download_file(bucket, csv_key, csv_path)
            df = pd.read_csv(csv_path)

            if df.empty:
                continue

            # Apply header only once (page 1)
            if header and len(header) == len(df.columns) and page_num == 1:
                df.columns = header

            dfs.append(df)
            logger.info(f"Added {len(df)} rows from page {page_num}")

        except Exception as e:
            logger.error(f"Failed to process page {page_num}: {e}")

    return (
        pd.concat(dfs, ignore_index=True).dropna(how="all").fillna("") if dfs else None
    )


def save_csv_to_s3(df, bucket, s3_key):
    """Save DataFrame as CSV to S3."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        upload_file(tmp.name, bucket, s3_key)
        os.unlink(tmp.name)


def send_notification(
    user_email, pdf_name, page_count, download_url, job_id, length_of_pdf
):
    """Send email notification and update DynamoDB."""
    if not user_email:
        return

    try:
        EmailService().send_processing_complete_notification(
            user_email=user_email,
            pdf_name=pdf_name,
            page_count=page_count,
            download_url=download_url,
        )

        # Mark email as sent for all records
        for i in range(length_of_pdf + 1):
            update_record_table(
                job_id=job_id,
                page_number=i,
                update_data={
                    "is_email_sent": True,
                    "updated_at": datetime.utcnow().isoformat(),
                    "status": "EMAIL_SENT",
                },
            )

        logger.info(f"Email sent to {user_email}")

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
