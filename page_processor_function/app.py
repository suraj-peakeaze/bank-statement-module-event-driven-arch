import tempfile
import os, base64, time, sys, json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


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
from utils.s3_utils import download_file, upload_file, send_sqs_message
from utils.check_completion import check_if_completed, check_if_email_sent
from utils.dynamo_utils import (
    update_record_table,
    get_items_from_record_table,
)
from utils.trigger_event import put_in_bridge
from utils.logger_config import setup_logging, get_logger


def lambda_handler(event, context):
    job_id = event.get("job_id", "unknown")
    page_number = event.get("page_number", "NA")

    # Initialize logging (runs once per Lambda execution)
    log_file = setup_logging(job_id, page_number)
    logger = get_logger(__name__)
    page_s3_key = None
    bucket = None
    item = None
    retries = 0
    user_email = ""
    pdf_name = ""
    length_of_pdf = ""
    table = ""

    raw_csv_s3_key = ""
    xml_s3_key = ""
    cleaned_s3_key = ""
    image_s3_key = ""

    try:
        # Handle SQS wrapper
        if "Records" in event:
            body = json.loads(event["Records"][0]["body"])
        else:
            body = event

        logger.info(f"body received: {body}")

        bucket = body.get("bucket") or os.getenv("PROCESSING_BUCKET")
        user_email = body.get("user_email")
        pdf_name = body.get("pdf_name")
        length_of_pdf = body.get("length_of_pdf")
        table = body.get("table")
        job_id = body.get("job_id")
        job_id = int(job_id)
        page_number = body.get("page_number")

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
        page_s3_key = item.get("page_s3_key")
        retries = int(item.get("retries", 0))
        page_length = item.get("length_of_pdf")
        page_length = int(page_length)

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

            update_record_table(
                job_id=job_id,
                page_number=page_number,
                update_data={
                    "calling_gemini_for_page_validation": True,
                },
            )

            # Extract Headers
            is_valid_page = extract_header_using_gemini(image=image_path)

            update_record_table(
                job_id=job_id,
                page_number=page_number,
                update_data={
                    "gemini_page_validation_completed": True,
                },
            )

            if is_valid_page:

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

                update_record_table(
                    job_id=job_id,
                    page_number=page_number,
                    update_data={
                        "calling_gemini_cleaning_process": True,
                    },
                )

                gemini_agent = GeminiAgent()
                gemini_response = gemini_agent.call_gemini(
                    xml_data_from_s3, downloaded_image_path
                )

                update_record_table(
                    job_id=job_id,
                    page_number=page_number,
                    update_data={
                        "gemini_cleaning_process_completed": True,
                    },
                )

                logger.info(
                    f"Got response from gemini in {type(gemini_response)} format, and final response is \n {gemini_response}"
                )

                cleaned_csv_path = os.path.join(
                    temp_dir, f"page_{page_number}_cleaned.csv"
                )
                process_gemini_response(gemini_response, cleaned_csv_path)

                if os.path.exists(cleaned_csv_path):
                    try:
                        cleaned_df = pd.read_csv(cleaned_csv_path)
                        logger.info(f"Final CSV before upload:\n {cleaned_df}")
                        if not cleaned_df.empty:
                            cleaned_s3_key = (
                                f"cleaned_csvs/{job_id}/page_{page_number}.csv"
                            )
                            upload_file(cleaned_csv_path, bucket, cleaned_s3_key)
                    except Exception as e:
                        logger.warning(f"Error reading cleaned CSV: {e}")

        print(
            f"""
                "page_number": {page_number},
                "status": "COMPLETED",
                "updated_at": {datetime.utcnow().isoformat()},
                "cleaned_csv_key": {cleaned_s3_key},
                "xml_key": {xml_s3_key},
                "raw_csv_key": {raw_csv_s3_key},
                "image_key": {image_s3_key},
                "retries": 0,
                "processor": "azure+gemini pipeline",
                "user_email": {item.get("user_email", "")},
                "next": "final_aggregator",
            """
        )

        update_record_table(
            job_id=job_id,
            page_number=page_number,
            update_data={
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
                "is_valid": is_valid_page,
            },
        )

        # Check if all pages are completed and trigger email if needed
        logger.info(f"Starting aggregation check for job_id: {job_id}")
        logger.info(f"page_length: {page_length}")

        try:
            # Small delay to ensure database consistency
            time.sleep(0.1)

            # Check each page completion status
            completion_count = 0
            for p in range(1, page_length + 1):
                is_completed = check_if_completed(job_id, p)
                if is_completed:
                    completion_count += 1
                logger.info(f"Page {p} completed: {is_completed}")

            logger.info(f"Completion count: {completion_count}/{page_length}")

            # Only trigger email if all pages are done
            if completion_count == page_length:
                email_sent = check_if_email_sent(job_id)
                logger.info(f"Email already sent: {email_sent}")

                if not email_sent:
                    logger.info(" All conditions met - triggering final aggregation")
                    result = {
                        "job_id": job_id,
                        "page_number": 0,
                        "bucket": bucket,
                        "table": table,
                    }

                    # Trigger the email
                    success = put_in_bridge(result)
                    logger.info(f"Bridge call result: {success}")
                else:
                    logger.info("All pages completed but email already sent")
            else:
                logger.info(
                    f"Not all pages completed yet: {completion_count}/{page_length}"
                )

        except Exception as e:
            logger.error(f"Error in aggregation trigger: {str(e)}")
            # Don't fail the whole function - the page was processed successfully

        # SUCCESS - Return successful response
        # Upload image to S3
        log_s3_key = (
            f"logs/{job_id}/azure+gemini_pipeline/log_{page_number}/{retries}.log"
        )
        upload_file(log_file, bucket, log_s3_key)

        logger.info(f"Page {page_number} processing completed successfully")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Page {page_number} processed successfully",
                    "job_id": job_id,
                    "page_number": page_number,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error processing page {page_number}: {str(e)}")
        retries += 1
        # SUCCESS - Return successful response# Upload image to S3
        log_s3_key = f"logs/{job_id}/log_{page_number}.log"
        upload_file(log_file, bucket, log_s3_key)

        logger.info(f"Page {page_number} processing completed successfully")

        current = get_items_from_record_table(job_id, page_number)
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
    raise
