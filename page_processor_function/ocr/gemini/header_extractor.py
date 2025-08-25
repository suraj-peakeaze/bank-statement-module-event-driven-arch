from typing import List
from google import genai
from google.genai import types
from pydantic import BaseModel
import os, base64
import json
from PIL import Image
from utils.logger_config import get_logger

logger = get_logger(__name__)

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from langfuse import get_client

# Environment variables
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Initialize Langfuse client and verify connectivity
langfuse = get_client()
assert langfuse.auth_check()


# Initialize OpenTelemetry ONCE at module level
endpoint = os.getenv("LANGFUSE_HOST") + "/api/public/otel/v1/traces"
auth = base64.b64encode(
    f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
).decode()

exporter = OTLPSpanExporter(
    endpoint=endpoint, headers={"Authorization": f"Basic {auth}"}
)

provider = TracerProvider(
    resource=Resource.create({"service.name": "header-extraction-service"})
)
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Initialize instrumentation ONCE
GoogleGenAIInstrumentor().instrument()

# Get tracer instance
tracer = trace.get_tracer(__name__)


class Response(BaseModel):
    step_by_step_reasoning: List[str]
    is_valid_page: bool


def prompt():
    return f"""
    Task:
        Analyze the provided bank statement image and tell me if the image contains actual transaction table or not, if not, mark is_valid_page as False, else mark it as True. To test if a page is actual bank statement or not, check if the page contains any transaction data like date, description, debit, credit, if it does, then it is a valid page, else it is not a valid page.
        
        Note:
            - Ignore all type of meta data tables like charge sheet, transaction summary, account details, etc.
            - Ignore all type of tables that are not transaction tables.
            - Ignore all types of tables that contains some type of details related to transaction history, bank's charges, etc.
            - We want to extract headers that are used to denote the columns related to a transaction in a bank account, we do not want to extract any other type of headers.
            - For eg, if you find a table that contains "transaction charges, this signifies what bank charges are for a transaction, "volume, this can be anything like number of transactions, "price, this can be anything like net amount transferred during a time period, "charges, this can be charges that bank charges per transaction, etc. then these headers denote data that is related to bank's service not user's transaction. A transaction table will have headers that will signify "date of transaction, i.e. when the transaction was done", "description of transaction, i.e. what was the transaction about", "debit amount", "credit amount", etc.
        
        Instructions:
            - Fields like "transaction charges", "transaction summary", "account details", "charge sheet", "bank's charges", etc. are not headers, ignore them and anything related to them.
    """


def extract_header_using_gemini(image):
    """
    Extract headers from bank statement image using Gemini API with proper Langfuse tracing

    Args:
        image: Image file path or PIL Image object

    Returns:
        tuple: (is_valid_page)
    """
    # Start a root span for the entire header extraction process
    with tracer.start_as_current_span("header_extraction_request") as root_span:
        logger.info(f"Image type: {type(image)}")
        # Set Langfuse trace attributes
        root_span.set_attribute("langfuse.trace.name", "header-extraction")
        root_span.set_attribute("langfuse.user.id", "user-123")
        root_span.set_attribute("service.name", "header-extraction-service")
        root_span.set_attribute("operation", "validate_page")

        try:
            # Initialize Gemini client
            with tracer.start_as_current_span("initialize_client") as client_span:
                client = genai.Client(
                    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GeminiApiKey")
                )
                model = "gemini-2.5-flash"
                client_span.set_attribute("model", model)

            # Process image
            with tracer.start_as_current_span("process_image") as image_span:
                if isinstance(image, str):
                    processed_image = Image.open(image)
                    image_span.set_attribute("image.source", "file_path")
                else:
                    processed_image = image
                    image_span.set_attribute("image.source", "pil_object")

                image_span.set_attribute(
                    "image.size", f"{processed_image.size[0]}x{processed_image.size[1]}"
                )
                image_span.set_attribute("image.mode", processed_image.mode)

            # Prepare content
            with tracer.start_as_current_span("prepare_content") as content_span:
                prompt_text = prompt()
                content = [prompt_text, processed_image]
                content_span.set_attribute("prompt.length", len(prompt_text))
                logger.info(f"Content prepared with prompt length: {len(prompt_text)}")

            # Make Gemini API call
            with tracer.start_as_current_span("gemini_api_call") as api_span:
                api_span.set_attribute("gen_ai.request.model", model)
                api_span.set_attribute("gen_ai.system.name", "gemini")
                api_span.set_attribute("gen_ai.operation.name", "generate_content")
                api_span.set_attribute("gen_ai.request.response_format", "json")

                # Add truncated prompt for observability
                truncated_prompt = (
                    prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text
                )
                api_span.set_attribute("gen_ai.prompt", truncated_prompt)

                response = client.models.generate_content(
                    model=model,
                    contents=content,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Response,
                    ),
                )

                # Set response attributes
                api_span.set_attribute("gen_ai.response.model", model)
                if hasattr(response, "usage_metadata"):
                    api_span.set_attribute(
                        "gen_ai.usage.input_tokens",
                        getattr(response.usage_metadata, "prompt_token_count", 0),
                    )
                    api_span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        getattr(response.usage_metadata, "candidates_token_count", 0),
                    )

            # Parse response
            with tracer.start_as_current_span("parse_response") as parse_span:
                try:
                    response_data = json.loads(response.text)
                    is_valid_page = response_data["is_valid_page"]
                    step_by_step_reasoning = response_data.get(
                        "step_by_step_reasoning", []
                    )

                    parse_span.set_attribute("parsing.status", "success")
                    parse_span.set_attribute("is_valid_page", is_valid_page)
                    parse_span.set_attribute(
                        "reasoning.steps", len(step_by_step_reasoning)
                    )

                    # Set completion attribute with truncated response
                    truncated_response = (
                        response.text[:1000] + "..."
                        if len(response.text) > 1000
                        else response.text
                    )
                    api_span.set_attribute("gen_ai.completion", truncated_response)

                    root_span.set_attribute("status", "success")
                    root_span.set_attribute("result.is_valid_page", is_valid_page)

                    return is_valid_page

                except (json.JSONDecodeError, KeyError) as e:
                    parse_span.set_attribute("parsing.status", "error")
                    parse_span.set_attribute("parsing.error", str(e))
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {response.text}")
                    raise

        except Exception as e:
            root_span.set_attribute("status", "error")
            root_span.set_attribute("error.message", str(e))
            root_span.set_attribute("error.type", type(e).__name__)
            logger.error(f"Error in extract_header_using_gemini: {e}")
            raise

        finally:
            # Ensure trace is flushed
            provider.force_flush(timeout_millis=5000)


# Alternative function with additional error handling and retry logic
def extract_header_with_retry(image, max_retries=3):
    """
    Validate page with retry logic and comprehensive error handling

    Args:
        image: Image file path or PIL Image object
        max_retries: Maximum number of retry attempts

    Returns:
        bool: is_valid_page
    """
    with tracer.start_as_current_span("header_extraction_with_retry") as retry_span:
        retry_span.set_attribute("langfuse.trace.name", "header-extraction-retry")
        retry_span.set_attribute("max_retries", max_retries)

        for attempt in range(max_retries):
            try:
                with tracer.start_as_current_span(
                    f"attempt_{attempt + 1}"
                ) as attempt_span:
                    attempt_span.set_attribute("attempt.number", attempt + 1)
                    is_valid_page = extract_header_using_gemini(image)

                    attempt_span.set_attribute("attempt.status", "success")
                    retry_span.set_attribute("successful_attempt", attempt + 1)

                    return is_valid_page

            except Exception as e:
                attempt_span.set_attribute("attempt.status", "failed")
                attempt_span.set_attribute("attempt.error", str(e))

                if attempt == max_retries - 1:
                    retry_span.set_attribute("final_status", "failed")
                    retry_span.set_attribute("final_error", str(e))
                    logger.error(f"All {max_retries} attempts failed. Final error: {e}")
                    raise

                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")

        return [], False
