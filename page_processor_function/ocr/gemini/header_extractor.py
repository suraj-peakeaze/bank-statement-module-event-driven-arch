from typing import List
from google import genai
from google.genai import types
from pydantic import BaseModel
import os
import json
from PIL import Image
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Header(BaseModel):
    index: int
    headers: str


class Response(BaseModel):
    headers: List[Header]
    step_by_step_reasoning: List[str]
    is_valid_page: bool


def prompt():
    return f"""
    Task:
        Analyze the provided bank statement image and extract the column headers that will be used for OCR processing. Return the headers as a list in the exact order they appear in the image from left to right.
        
        Also, analyse the image and tell me if the image contains actual transaction table or not, if not, return empty list and mark is_valid_page as False, else mark it as True. to test if a page is actual bank statement or not, check if the page contains any transaction data like date, description, debit, credit, if it does, then it is a valid page, else it is not a valid page.
        
        Note:
            - Ignore all type of meta data tables like charge sheet, transaction summary, account details, etc.
            - Ignore all type of tables that are not transaction tables.
            - Ignore all types of tables that contains some type of details related to transaction history, bank's charges, etc.
            - We want to extract headers that are used to denote the columns related to a transaction in a bank account, we do not want to extract any other type of headers.
            - For eg, if you find a table that contains "transaction charges, this signifies what bank charges are for a transaction, "volume, this can be anything like number of transactions, "price, this can be anything like net amount transferred during a time period, "charges, this can be charges that bank charges per transaction, etc. then these headers denote data that is related to bank's service not user's transaction. A transaction table will have headers that will signify "date of transaction, i.e. when the transaction was done", "description of transaction, i.e. what was the transaction about", "debit amount", "credit amount", etc.
        
        Instructions:
            - Fields like "transaction charges", "transaction summary", "account details", "charge sheet", "bank's charges", etc. are not headers, ignore them and anything related to them.
            - Identify table headers: Look for the row containing column labels/headers in the bank statement.
            - Extract in order: List headers in the exact sequence they appear (left to right).
            - Return empty list: If no clear headers are visible or identifiable, return [].
            - Extract all headers: Extract all headers exactly as they appear in the image.
            - Return empty list if you are not able to extract the headers.
    """


def extract_header_using_gemini(image):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GeminiApiKey"))
    model = "gemini-2.0-flash-001"

    image = Image.open(image)
    content = [prompt(), image]
    logger.info(f"content: {content}")

    response = client.models.generate_content(
        model=model,
        contents=content,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Response,
        ),
    )

    response = json.loads(response.text)
    header = response["headers"]
    is_valid_page = response["is_valid_page"]
    logger.info(f"response: \n{response}")

    return header, is_valid_page
