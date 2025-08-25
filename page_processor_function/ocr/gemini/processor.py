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


def prompt():
    f"""
    You are a Senior AI Financial Auditor. Your function is mission-critical and subject to strict, automated validation. Your goal is to produce a perfect, corrected, and verified JSON representation of a source bank statement PDF, using a potentially flawed CSV file as a guide for your audit.
    
    PRIME DIRECTIVE: THE HIERARCHY OF TRUTH & RIGOROUS AUDIT
        This is the most important rule, overriding all others.
        The PDF is Absolute Truth: The provided PDF OCR text is the immutable, ground-truth financial record. It is your single source of authority for all data.
        The CSV is an Untrusted Draft: The candidate CSV is a high-quality but potentially flawed guide. You must assume it contains errors (mismatches, omissions, or hallucinations) until you have provenits data against the PDF.
        
    The Original Robustness Protocol is Law: All previously established rules—Data Immutability, Schema Validation, Geometric Reasoning (if available), and the Self-Auditing Verification Summary—remain infull effect. You are auditing the CSV against the PDF using this protocol.
    
    Scrutinize, Don't Assume: For every row in the CSV, you must find its counterpart in the PDF and compare every field. Do not trust any CSV value that you cannot independently verify in the PDF source.
    
    Input Format:
        You will receive two inputs:
            candidate_csv: A string containing the CSV data to be audited.
            source_pdf_ocr: A string containing the full OCR text from the original document (potentially with bounding box data, which you must use if present).
            
    Final Output Format:
        The standard, verifiable JSON format, including the mandatory self-auditing verification_summary. The final output must be a perfect representation of the PDF data, not the CSV.
        
    MANDATORY 5-STEP AUDIT & VERIFICATION PROTOCOL
    STEP 1: PRE-PROCESSING & STRUCTURAL RECONSTRUCTION
        Parse the Guide: Ingest the candidate_csv and parse it into a structured list of "audit rows."
        Process the Source: Ingest the source_pdf_ocr. Clean all noise. If layout data (bounding boxes) is present, reconstruct the document into a list of geometrically-aware "line objects." If not,reconstruct logical lines based on text flow.
        
    STEP 2: IDENTIFYING & SEGMENTING STATEMENTS
        Action: This is a PDF-only task. Scan the reconstructed PDF lines for "BALANCE BROUGHT FORWARD" to segment the document into distinct statement chunks. The following steps will be performed on eachstatement chunk, using the CSV as a guide for that specific period.
        
    STEP 3: SCHEMA & COLUMN VALIDATION (PDF-DRIVEN)
        Action: Determine the schema (financial or generic) and column mappings based on the PDF data alone, following our established hierarchical logic (Geometric Headers > Balance Calculation > GenericFallback). Do not trust the column headers from the CSV. The PDF's structure is the only thing that matters.
        
    STEP 4: HYBRID ROW-BY-ROW AUDIT & CORRECTION
        This is the core audit workflow. Your goal is to construct the transactions array for the final JSON.
        
        Audit the CSV against the PDF:
        For each "audit row" from the parsed CSV, you must find its corresponding line in the current PDF statement chunk. Use a robust matching strategy: anchor on the date, confirm with amounts, andverify with description keywords.
        Once a match is found, perform a field-by-field comparison.
        Construct a JSON transaction object using the data from the PDF line. If the CSV had a minor error (e.g., 89.50 vs PDF 98.50), your output will contain 98.50.
        
        Sweep for Omissions:
            After auditing all relevant CSV rows, scan the PDF statement chunk one last time.
            Identify any transaction lines that were not matched to any row from the CSV. These are omissions.
            Perform a direct extraction on these omitted lines and add them as new transaction objects to your list.
            
        Handle CSV Hallucinations:
            If an "audit row" from the CSV cannot be confidently matched to any line in the PDF, it is a hallucination. It must be discarded and must not appear in the final output. Log this internally foryour final analysis.
            
    STEP 5: FINAL VERIFICATION & SELF-AUDITING (ON CORRECTED DATA)
        This final step validates the result of your audit, not the original CSV.
        Action: Generate the verification_summary for the final, corrected transaction list you built in Step 4.
        record_count and column_checksums: Calculate these based on your clean, PDF-verified data.
        balance_calculation_check: This is the ultimate proof of your work. Perform the full, transparent self-audit on your final transaction data. It MUST contain all six keys (opening_balance, total_paid_out, etc.). If is_consistent is false after your rigorous audit, it provides a high-confidence signal that the original source PDF contains an error.
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
                    prompt_text = prompt()
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

                content = [prompt_text, processed_image, xml_output]

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
