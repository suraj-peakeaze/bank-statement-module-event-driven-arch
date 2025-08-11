import tempfile
import os
import sys
import pandas as pd
import logging
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append("/opt/python")

from pdf2image import convert_from_path
from ocr.azure_processor import AzureAgent
from ocr.gemini.header_extractor import extract_header_using_gemini
from ocr.gemini.processor import GeminiAgent
from ocr.gemini.cleaner import process_gemini_response
from utils.xml_utils import run_file_to_xml_converter
from utils.s3_utils import download_file, upload_file
from utils.check_completion import check_if_completed, check_if_email_sent
from utils.dynamo_utils import (
    update_record_table,
    get_items_from_record_table,
)
from utils.trigger_event import put_in_bridge


def lambda_handler(event, context):
    job_id = None
    page_number = None
    page_s3_key = None
    bucket = None
    item = None
    retries = 0

    raw_csv_s3_key = None
    xml_s3_key = None
    cleaned_s3_key = None
    image_s3_key = None

    try:
        # Handle SQS wrapper
        if "Records" in event:
            body = json.loads(event["Records"][0]["body"])
        else:
            body = event

        logger.info(f"body received: {body}")

        bucket = body.get("bucket") or os.getenv("PROCESSING_BUCKET")
        job_id = body.get("job_id")
        page_number = body.get("page_number")  # Correct page number usage

        if not bucket:
            raise ValueError(
                "bucket not provided in payload and PROCESSING_BUCKET not set"
            )
        if not job_id:
            raise ValueError("job_id not found in event")
        if page_number is None:
            raise ValueError("page_number not found in event")

        items = get_items_from_record_table(job_id=job_id, page_number=page_number)
        if not items:
            raise ValueError(
                f"No matching record found for job_id={job_id}, page_number={page_number}"
            )

        item = items[0]
        page_s3_key = item.get("pdf_key")
        retries = int(item.get("retries", 0))
        page_length = item.get("total_pages")

        logger.info(f"bucket: {bucket}")
        logger.info(f"page_number: {page_number}")
        logger.info(f"page_s3_key: {page_s3_key}")
        logger.info(f"Processing page {page_number} for job {job_id}")

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"[DEBUG] [APP] Created temp directory: {temp_dir}")

            # Download PDF page
            local_pdf = os.path.join(temp_dir, f"page_{page_number}.pdf")
            download_file(bucket, page_s3_key, local_pdf)
            logger.info(f"Downloaded PDF to {local_pdf}")

            # Convert PDF to image
            images = convert_from_path(local_pdf)
            if not images:
                raise Exception(f"Could not convert page {page_number} to image")

            image_path = os.path.join(temp_dir, f"page_{page_number}.png")
            images[0].save(image_path, "PNG")
            logger.info(f"Image saved at {image_path}")

            # Upload image to S3
            image_s3_key = f"images/{job_id}/page_{page_number}.png"
            upload_file(image_path, bucket, image_s3_key)

            # Extract Headers
            header, is_valid_page = extract_header_using_gemini(image=image_path)
            index_to_header = (
                {item["index"]: item["headers"] for item in header}
                if is_valid_page
                else {}
            )
            logger.info(
                f"Extracted column headers: {[item['headers'] for item in header]}"
            )

            # Azure OCR
            temp_csv = os.path.join(temp_dir, f"page_{page_number}_raw.csv")
            agent = AzureAgent(images, image_path, page_number, temp_path=temp_csv)
            result_df = agent.process_page()

            if not os.path.exists(temp_csv):
                logger.warning(f"Raw CSV not found: {temp_csv}")
                return

            raw_csv_s3_key = f"raw_csvs/{job_id}/page_{page_number}.csv"
            upload_file(temp_csv, bucket, raw_csv_s3_key)

            # Convert to XML
            xml_data = run_file_to_xml_converter(temp_csv, temp_dir)
            xml_path = os.path.join(temp_dir, f"page_{page_number}.xml")
            with open(xml_path, "w") as f:
                f.write(xml_data)
            xml_s3_key = f"xmls/{job_id}/page_{page_number}.xml"
            upload_file(xml_path, bucket, xml_s3_key)

            # Download again for cleaning
            s3_raw_csv = os.path.join(
                temp_dir, f"downloaded_page_{page_number}_raw.csv"
            )
            s3_xml = os.path.join(temp_dir, f"downloaded_page_{page_number}.xml")
            download_file(bucket, raw_csv_s3_key, s3_raw_csv)
            download_file(bucket, xml_s3_key, s3_xml)

            if not os.path.exists(s3_raw_csv) or not os.path.exists(s3_xml):
                raise FileNotFoundError("Downloaded files missing")

            with open(s3_xml, "r") as f:
                xml_data_from_s3 = f.read()

            downloaded_image_path = os.path.join(
                temp_dir, f"downloaded_page_{page_number}.png"
            )
            download_file(bucket, image_s3_key, downloaded_image_path)
            if not os.path.exists(downloaded_image_path):
                raise FileNotFoundError(
                    f"Downloaded image not found: {downloaded_image_path}"
                )

            gemini_agent = GeminiAgent()
            gemini_response = gemini_agent.call_gemini(
                xml_data_from_s3, downloaded_image_path, index_to_header
            )

            cleaned_csv = os.path.join(temp_dir, f"page_{page_number}_cleaned.csv")
            process_gemini_response(gemini_response, s3_raw_csv, cleaned_csv)

            if os.path.exists(cleaned_csv):
                try:
                    cleaned_df = pd.read_csv(cleaned_csv)
                    if not cleaned_df.empty:
                        cleaned_s3_key = f"cleaned_csvs/{job_id}/page_{page_number}.csv"
                        upload_file(cleaned_csv, bucket, cleaned_s3_key)
                except Exception as e:
                    logger.warning(f"Error reading cleaned CSV: {e}")

        # Success update
        success_item = {
            "page_number": page_number,
            "status": "COMPLETED",
            "updated_at": datetime.utcnow().isoformat(),
            "cleaned_csv_key": cleaned_s3_key,
            "xml_key": xml_s3_key,
            "raw_csv_key": raw_csv_s3_key,
            "image_key": image_s3_key,
            "retries": retries,
            "user_email": item.get("user_email", ""),
            "next": "final_aggregator",
            "header": header,
        }

        page_info = item.get("page_info", [])
        if not isinstance(page_info, list):
            page_info = []
        page_info = [p for p in page_info if p.get("page_number") != page_number]
        page_info.append(success_item)

        update_record_table(
            job_id=job_id,
            page_number=page_number,
            update_data={
                "page_info": page_info,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )

        # Final aggregation trigger
        try:
            completion_count = sum(
                1 for p in range(1, page_length + 1) if check_if_completed(job_id, p)
            )
            if completion_count == page_length and not check_if_email_sent(job_id):
                result = {"job_id": job_id, "page_number": 0}
                put_in_bridge(result)
        except Exception as e:
            raise Exception("Unable to complete process:", e)

        return {"job_id": job_id}

    except Exception as e:
        logger.error(f"Error processing page {page_number}: {str(e)}")
        retries += 1
        status = "UN_PROCESSABLE" if retries >= 3 else "FAILED"

        error_item = {"retries": retries, "status": status}
        if job_id and page_number is not None:
            try:
                update_record_table(
                    job_id=job_id, page_number=page_number, update_data=error_item
                )
            except Exception as db_error:
                logger.error(f"Failed to write error record: {str(db_error)}")
        raise e
