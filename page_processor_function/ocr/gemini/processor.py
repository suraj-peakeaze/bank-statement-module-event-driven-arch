import json
from google import genai
from google.genai import types
import os, base64
from utils.logger_config import get_logger

logger = get_logger(__name__)

from utils.pydantic_schema_handler import BankStatement
from PIL import Image

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from langfuse import get_client

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Initialize Langfuse client and verify connectivity
langfuse = get_client()
assert langfuse.auth_check()

# Initialize OpenTelemetry ONCE at module level, not inside each method call
endpoint = os.getenv("LANGFUSE_HOST") + "/api/public/otel/v1/traces"
auth = base64.b64encode(
    f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
).decode()

exporter = OTLPSpanExporter(
    endpoint=endpoint, headers={"Authorization": f"Basic {auth}"}
)

provider = TracerProvider(
    resource=Resource.create({"service.name": "bank-statement-processor"})
)
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Initialize instrumentation ONCE
GoogleGenAIInstrumentor().instrument()


def prompt(xml_output):
    f"""
    
    """


class GeminiAgent:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GeminiApiKey")
        )
        self.model = "gemini-2.5-pro"
        self.tracer = trace.get_tracer(__name__)

    def call_gemini(self, xml_output, image):
        """
        Call Gemini API with proper Langfuse tracing
        """
        # Start a root span for the entire request
        with self.tracer.start_as_current_span(
            "gemini_bank_statement_processing"
        ) as root_span:
            logger.info(f"Image type: {type(image)}")
            # Set Langfuse trace attributes
            root_span.set_attribute("langfuse.trace.name", "bank-statement-formatting")
            root_span.set_attribute("langfuse.user.id", "user-123")
            root_span.set_attribute("service.name", "BSSS-bank-statement-processor")

            try:
                # Prepare prompt
                with self.tracer.start_as_current_span("prepare_prompt") as prompt_span:
                    prompt_text = prompt(xml_output)
                    prompt_span.set_attribute("prompt.length", len(prompt_text))
                    logger.info(f"Prompt prepared with length: {len(prompt_text)}")

                # Process image
                with self.tracer.start_as_current_span("process_image") as image_span:
                    if isinstance(image, str):
                        processed_image = Image.open(image)
                    else:
                        processed_image = image
                    image_span.set_attribute(
                        "image.size",
                        f"{processed_image.size[0]}x{processed_image.size[1]}",
                    )
                    image_span.set_attribute("image.mode", processed_image.mode)

                content = [prompt_text, processed_image]

                # Make the Gemini API call
                with self.tracer.start_as_current_span("gemini_api_call") as api_span:
                    api_span.set_attribute("gen_ai.request.model", self.model)
                    api_span.set_attribute("gen_ai.system.name", "gemini")
                    api_span.set_attribute("gen_ai.operation.name", "generate_content")
                    api_span.set_attribute("gen_ai.request.temperature", 1)

                    # Add a truncated version of the prompt for observability
                    truncated_prompt = (
                        prompt_text[:1000] + "..."
                        if len(prompt_text) > 1000
                        else prompt_text
                    )
                    api_span.set_attribute("gen_ai.prompt", truncated_prompt)

                    response = self.client.models.generate_content(
                        model=self.model,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0,
                            response_schema=BankStatement,
                            thinking_config=types.ThinkingConfig(include_thoughts=True),
                        ),
                        contents=content,
                    )

                    # Set response attributes
                    api_span.set_attribute("gen_ai.response.model", self.model)
                    if hasattr(response, "usage_metadata"):
                        api_span.set_attribute(
                            "gen_ai.usage.input_tokens",
                            getattr(response.usage_metadata, "prompt_token_count", 0),
                        )
                        api_span.set_attribute(
                            "gen_ai.usage.output_tokens",
                            getattr(
                                response.usage_metadata, "candidates_token_count", 0
                            ),
                        )

                # Process response
                with self.tracer.start_as_current_span(
                    "process_response"
                ) as response_span:
                    thoughts = ""
                    answer = ""

                    for part in response.candidates[0].content.parts:
                        if not part.text:
                            continue
                        elif part.thought:
                            if not thoughts:
                                logger.info("Processing thoughts...")
                            thoughts += part.text
                        else:
                            if not answer:
                                logger.info("Processing answer...")
                            answer += part.text

                    response_span.set_attribute("thoughts", thoughts)
                    response_span.set_attribute("answer", answer)

                    # Set the completion attribute with truncated response for observability
                    truncated_response = (
                        response.text[:1000] + "..."
                        if len(response.text) > 1000
                        else response.text
                    )
                    api_span.set_attribute("gen_ai.completion", truncated_response)

                # Parse JSON response - Return full bank statement
                with self.tracer.start_as_current_span("parse_json") as json_span:
                    try:
                        logger.info(f"Response received: {response.text}")
                        parsed_response = json.loads(response.text)

                        transactions = []
                        schema_type = "unknown"

                        if "financial_transactions" in parsed_response:
                            # ✅ New schema
                            transactions = parsed_response.get(
                                "financial_transactions", []
                            )
                            schema_type = parsed_response.get(
                                "schema_type", "financial"
                            )

                        elif "generic_transactions" in parsed_response:
                            # ✅ Fallback schema
                            transactions = parsed_response.get(
                                "generic_transactions", []
                            )
                            schema_type = parsed_response.get("schema_type", "generic")

                        if transactions:
                            json_span.set_attribute(
                                "transactions.count", len(transactions)
                            )
                            json_span.set_attribute("schema_type", schema_type)
                            json_span.set_attribute("parsing.status", "success")

                            logger.info(
                                f"Successfully parsed response with {len(transactions)} transactions (schema: {schema_type})"
                            )
                            root_span.set_attribute("status", "success")
                            return transactions

                        # No transactions at all
                        logger.warning("No transactions found in response")
                        json_span.set_attribute("parsing.status", "no_data")
                        return None

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        json_span.set_attribute("parsing.status", "error")
                        json_span.set_attribute("parsing.error", str(e))
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response text: {response.text}")
                        raise

            except Exception as e:
                root_span.set_attribute("status", "error")
                root_span.set_attribute("error.message", str(e))
                root_span.set_attribute("error.type", type(e).__name__)
                logger.error(f"Error in call_gemini: {e}")
                raise
