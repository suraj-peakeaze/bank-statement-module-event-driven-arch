import pandas as pd
from utils.logger_config import get_logger

logger = get_logger(__name__)


def process_gemini_response(gemini_response, extracted_data_path):
    logger.info(f"Processing Gemini response of type: {type(gemini_response)}")

    # Handle list response format (direct list of transactions)
    if isinstance(gemini_response, list):
        logger.info(
            f"Got response from gemini in {type(gemini_response)} format, and final response is \n {gemini_response}"
        )
        transactions = gemini_response
        if not transactions:
            raise ValueError("No transactions found in Gemini response (list format)")

    # Handle dictionary response format (with nested transaction keys)
    elif isinstance(gemini_response, dict):
        logger.info(f"Finding transactions from response: \n {gemini_response}")
        transactions = gemini_response.get("financial_transactions", [])
        logger.info(f"Got transactions from financial_transactions: {transactions}")
        if not transactions:
            transactions = gemini_response.get("generic_transactions", [])
        logger.info(f"Got transactions from generic_transactions: {transactions}")

        if not transactions:
            raise ValueError("No transactions found in Gemini response (dict format)")

    else:
        raise ValueError(f"Unexpected response format: {type(gemini_response)}")

    # Normalize to DataFrame
    df = pd.json_normalize(transactions)
    df = df.replace(r"\n", " ", regex=True)
    df.to_csv(extracted_data_path, index=False)
    return df
