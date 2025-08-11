from datetime import datetime
import boto3
import json
import pandas as pd
import tempfile
import os
import sys
from dotenv import load_dotenv
import logging

# Load env vars early
load_dotenv()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add parent directories to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.append("/opt/python")
sys.path.append(os.path.join(BASE_DIR, "../../"))

from utils.s3_utils import download_file, upload_file
from utils.email_service import EmailService
from utils.dynamo_utils import get_items_from_record_table, update_record_table

s3_client = boto3.client("s3")


def lambda_handler(event, context):
    """Lambda function to aggregate all processed pages for a given job."""
    try:
        # --- Event parsing ---
        body = _extract_event_body(event)
        job_id = body["job_id"]
        bucket = body.get("bucket") or os.getenv("PROCESSING_BUCKET")

        if not bucket:
            raise ValueError("No S3 bucket provided or configured")

        # --- Fetch metadata from DynamoDB ---
        items = get_items_from_record_table(job_id=job_id, page_number=0)
        item = items[0]
        length_of_pdf = item.get("length_of_pdf")
        if not item:
            raise ValueError(f"No record found for job_id {job_id}")

        user_email = item.get("user_email")
        pdf_name = item.get("pdf_name", f"{job_id}_output")
        header = item.get("header")
        page_info = item.get("page_info", [])

        logger.info(f"Processing job {job_id} with {len(page_info)} page(s)")

        if not isinstance(page_info, list) or not page_info:
            logger.warning(f"No valid page_info found for job_id {job_id}")
            return _build_result(job_id, "FAILED", "No page_info found")

        # --- Sort results by page number ---
        sorted_results = sorted(
            [p for p in page_info if isinstance(p, dict)],
            key=lambda x: x.get("page_number", 0),
        )

        # --- Merge CSVs ---
        final_df = _merge_page_csvs(sorted_results, bucket, header, pdf_name)

        if final_df is None or final_df.empty:
            logger.warning(f"No valid CSV data found for job_id {job_id}")
            return _build_result(job_id, "FAILED", "No valid CSV data found")

        # --- Save final CSV to S3 ---
        with tempfile.TemporaryDirectory() as temp_dir:
            final_csv_path = os.path.join(temp_dir, f"{pdf_name}_final.csv")
            final_df.to_csv(final_csv_path, index=False)
            final_s3_key = f"final_outputs/{job_id}/{pdf_name}_final.csv"
            upload_file(final_csv_path, bucket, final_s3_key)

        # --- Generate presigned URL ---
        download_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": final_s3_key},
            ExpiresIn=3600,  # 1 hour
        )

        # --- Send email notification ---
        if user_email:
            try:
                EmailService().send_processing_complete_notification(
                    user_email=user_email,
                    pdf_name=pdf_name,
                    page_count=len(sorted_results),
                    download_url=download_url,
                )

                # Update DynamoDB record to indicate email was sent
                for i in range(length_of_pdf + 1):
                    update_record_table(
                        page_id=job_id,
                        page_number=i,
                        update_data={
                            "is_email_sent": True,
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                    )

            except Exception as e:
                logger.error(f"Failed to send email to {user_email}: {e}")

        logger.info(f"Job {job_id} aggregation completed successfully")
        return _build_result(
            job_id, "COMPLETED", None, final_s3_key, download_url, len(final_df)
        )

    except Exception as e:
        logger.exception(f"Error in aggregator: {e}")
        raise


# --- Helper functions ---


def _extract_event_body(event):
    """Extracts and parses the event body from direct or SQS-triggered invocation."""
    if "Records" in event:  # SQS
        return json.loads(event["Records"][0]["body"])
    return event


def _merge_page_csvs(sorted_results, bucket, header, pdf_name):
    """Downloads, validates, and merges CSV files from each page."""
    dfs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for result in sorted_results:
            csv_key = result.get("cleaned_csv_key")
            page_num = result.get("page_number")

            if not csv_key:
                logger.info(f"Skipping page {page_num} - no transactions found")
                continue

            csv_path = os.path.join(temp_dir, f"page_{page_num}.csv")
            download_file(bucket, csv_key, csv_path)

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    logger.info(f"Page {page_num} CSV is empty")
                    continue

                if header and len(header) == len(df.columns):
                    df.columns = header
                elif header:
                    logger.warning(
                        f"Page {page_num} header length mismatch "
                        f"({len(header)} vs {len(df.columns)})"
                    )

                dfs.append(df)
                logger.info(f"Added {len(df)} rows from page {page_num}")

            except Exception as e:
                logger.error(f"Could not read CSV for page {page_num}: {e}")

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True).dropna(how="all").fillna("")


def _build_result(
    job_id,
    status,
    error_message=None,
    final_csv_key=None,
    download_url=None,
    record_count=0,
):
    """Builds the final Lambda response."""
    result = {"job_id": job_id, "status": status, "record_count": record_count}
    if error_message:
        result["error"] = error_message
    if final_csv_key:
        result["final_csv_key"] = final_csv_key
    if download_url:
        result["download_url"] = download_url
    return result
