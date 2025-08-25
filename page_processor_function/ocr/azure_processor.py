from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
import pandas as pd
from utils.logger_config import get_logger

logger = get_logger(__name__)
# Azure credentials
endpoint = os.getenv("AZURE_END_POINT") or os.getenv("AzureEndPoint")
key = os.getenv("AZURE_SECRET_KEY") or os.getenv("AzureApiKey")

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)


def make_serializable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    elif hasattr(value, "text"):
        return value.text
    elif hasattr(value, "value"):
        return value.value
    elif isinstance(value, list):
        return [make_serializable(item) for item in value]
    elif hasattr(value, "amount") and hasattr(value, "currency_symbol"):
        return f"{value.currency_symbol}{value.amount}"
    else:
        return str(value)


def extract_table_data(tables):
    """
    Extract cell content, row and column indices from Azure Document Intelligence tables response.

    Args:
        tables: List of table objects from Azure response

    Returns:
        List of dictionaries containing extracted table data
    """
    extracted_tables = []

    for table_idx, table in enumerate(tables):
        try:
            # Handle both Azure response objects and dictionary formats
            if hasattr(table, "row_count") and hasattr(table, "column_count"):
                # Azure response object format (current SDK)
                row_count = table.row_count
                column_count = table.column_count
                cells = table.cells
                logger.info(
                    f"[DEBUG] Table {table_idx} - Azure object format: {row_count}x{column_count}"
                )
            elif hasattr(table, "rowCount") and hasattr(table, "columnCount"):
                # Alternative Azure response object format
                row_count = table.rowCount
                column_count = table.columnCount
                cells = table.cells
                logger.info(
                    f"[DEBUG] Table {table_idx} - Azure object alt format: {row_count}x{column_count}"
                )
            elif isinstance(table, dict):
                # Dictionary format (serialized response)
                row_count = table.get("rowCount", 0)
                column_count = table.get("columnCount", 0)
                cells = table.get("cells", [])
                logger.info(
                    f"[DEBUG] Table {table_idx} - Dictionary format: {row_count}x{column_count}"
                )
            else:
                # Log all available attributes for debugging
                available_attrs = [
                    attr for attr in dir(table) if not attr.startswith("_")
                ]
                logger.warning(
                    f"Unknown table format at index {table_idx}: {type(table)}, "
                    f"Available attributes: {available_attrs}"
                )
                continue

            table_info = {
                "table_index": table_idx,
                "row_count": row_count,
                "column_count": column_count,
                "cells": [],
                "structured_data": {},
            }

            # Extract individual cells
            for cell in cells:
                # Handle both Azure response objects and dictionary formats
                if hasattr(cell, "row_index") and hasattr(cell, "column_index"):
                    # Azure response object format (current SDK)
                    cell_data = {
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "content": cell.content.strip() if cell.content else "",
                        "kind": getattr(cell, "kind", ""),
                        "bounding_regions": getattr(cell, "bounding_regions", []),
                        "spans": getattr(cell, "spans", []),
                    }
                elif hasattr(cell, "rowIndex") and hasattr(cell, "columnIndex"):
                    # Alternative Azure response object format
                    cell_data = {
                        "row_index": cell.rowIndex,
                        "column_index": cell.columnIndex,
                        "content": cell.content.strip() if cell.content else "",
                        "kind": getattr(cell, "kind", ""),
                        "bounding_regions": getattr(cell, "bounding_regions", []),
                        "spans": getattr(cell, "spans", []),
                    }
                elif isinstance(cell, dict):
                    # Dictionary format
                    cell_data = {
                        "row_index": cell.get("rowIndex"),
                        "column_index": cell.get("columnIndex"),
                        "content": cell.get("content", "").strip(),
                        "kind": cell.get("kind", ""),
                        "bounding_regions": cell.get("boundingRegions", []),
                        "spans": cell.get("spans", []),
                    }
                else:
                    # Log available attributes for debugging
                    cell_attrs = [
                        attr for attr in dir(cell) if not attr.startswith("_")
                    ]
                    logger.warning(
                        f"Unknown cell format in table {table_idx}: {type(cell)}, "
                        f"Available attributes: {cell_attrs[:10]}"
                    )
                    continue

                table_info["cells"].append(cell_data)

            # Create structured representation (matrix format)
            if table_info["cells"]:
                max_row = max(
                    [
                        cell["row_index"]
                        for cell in table_info["cells"]
                        if cell["row_index"] is not None
                    ]
                )
                max_col = max(
                    [
                        cell["column_index"]
                        for cell in table_info["cells"]
                        if cell["column_index"] is not None
                    ]
                )
            else:
                max_row = max_col = 0

            # Initialize matrix
            matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            headers = ["" for _ in range(max_col + 1)]

            # Fill matrix and extract headers
            for cell in table_info["cells"]:
                row_idx = cell["row_index"] or 0
                col_idx = cell["column_index"] or 0
                content = cell["content"]

                if row_idx < len(matrix) and col_idx < len(matrix[row_idx]):
                    matrix[row_idx][col_idx] = content

                # Store column headers
                if cell["kind"] == "columnHeader" and col_idx < len(headers):
                    headers[col_idx] = content

            table_info["structured_data"] = {
                "headers": headers,
                "matrix": matrix,
                "rows": matrix[1:] if len(matrix) > 1 else [],  # Exclude header row
            }

            extracted_tables.append(table_info)

            # Log extracted data for debugging
            logger.info(
                f"[DEBUG] [EXTRACT_TABLE_DATA] Table {table_idx}: {table_info['row_count']}x{table_info['column_count']}"
            )
            logger.info(f"[DEBUG] [EXTRACT_TABLE_DATA] Headers: {headers}")
            logger.info(
                f"[DEBUG] [EXTRACT_TABLE_DATA] Total cells: {len(table_info['cells'])}"
            )

        except Exception as e:
            logger.info(f"Error processing table {table_idx}: {e}")
            continue

    return extracted_tables


class AzureAgent:
    def __init__(self, images, page_path, page_num, retries=3, temp_path=None):
        self.images = images
        self.retries = retries
        self.combined_data = []
        self.page_path = page_path
        self.page_num = page_num
        self.temp_path = temp_path
        logger.info(
            f"[DEBUG] [AZURE_AGENT] Initialized with temp_path: {self.temp_path}"
        )
        logger.info(f"[DEBUG] [AZURE_AGENT] page_path: {self.page_path}")
        logger.info(f"[DEBUG] [AZURE_AGENT] page_num: {self.page_num}")

    def process_page(self):
        page_data = {
            "page_number": self.page_num,
            "file_path": self.page_path,
            "analysis": {},
            "extracted_tables": [],
        }

        df = None  # Initialize df outside the try block
        attempt = 0
        while attempt < self.retries:
            try:
                logger.info(f"[DEBUG] [PROCESS_PAGE] page_path: {self.page_path}")
                logger.info(
                    f"[DEBUG] [PROCESS_PAGE] File exists: {os.path.exists(self.page_path)}"
                )
                logger.info(
                    f"[DEBUG] [PROCESS_PAGE] File size: {os.path.getsize(self.page_path) if os.path.exists(self.page_path) else 'N/A'}"
                )
                with open(self.page_path, "rb") as file:
                    poller = document_intelligence_client.begin_analyze_document(
                        "prebuilt-invoice", file
                    )
                    result = poller.result()
                    logger.info(
                        f"[DEBUG] [PROCESS_PAGE] result.tables: {result.tables}"
                    )
                    # Extract table data if tables exist
                    if hasattr(result, "tables") and result.tables:
                        # Add debugging for table structure analysis
                        try:
                            extracted_tables = extract_table_data(result.tables)
                            page_data["extracted_tables"] = extracted_tables
                            matrix = []
                            headers = []
                            df_list = []
                            df = pd.DataFrame()
                            # Log sample of extracted data
                            for table in extracted_tables:
                                try:
                                    headers = table["structured_data"]["headers"]
                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] Extracted table headers: {headers}"
                                    )
                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] Sample rows: {table['structured_data']['rows']}"
                                    )
                                    buffer_matrix = table["structured_data"]["matrix"]
                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] matrix: {buffer_matrix}"
                                    )
                                    # Create DataFrame with robust error handling
                                    try:
                                        matrix = (
                                            matrix + [headers] + buffer_matrix
                                        )  # add headers to the row 0 of the dataframe
                                        temp_df = pd.DataFrame(matrix)
                                        logger.info(
                                            f"[DEBUG] [PROCESS_PAGE] DataFrame with headers at row 0: {df}"
                                        )
                                        if not temp_df.empty:
                                            # Clean DataFrame content
                                            temp_df = temp_df.map(
                                                lambda x: (
                                                    x.replace("\n", " ")
                                                    .replace("?", "")
                                                    .replace(":selected:", "")
                                                    .replace(":unselected:", "")
                                                    .strip()
                                                    if isinstance(x, str)
                                                    else x
                                                )
                                            )
                                            df_list.append(temp_df)
                                            logger.info(
                                                f"[DEBUG] [PROCESS_PAGE] DataFrame for page {self.page_num}:\n{df}"
                                            )
                                            logger.info(
                                                f"[DEBUG] [PROCESS_PAGE] DataFrame shape: {df.shape}"
                                            )
                                            logger.info(
                                                f"[DEBUG] [PROCESS_PAGE] DataFrame dtypes:\n{df.dtypes}"
                                            )
                                        else:
                                            logger.warning(
                                                f"[WARNING] [PROCESS_PAGE] Empty DataFrame created for page {self.page_num}"
                                            )
                                    except Exception as df_error:
                                        logger.error(
                                            f"[ERROR] [PROCESS_PAGE] Failed to create or process DataFrame for page {self.page_num}: {df_error}"
                                        )
                                        df = None
                                except Exception as table_processing_error:
                                    logger.error(
                                        f"[ERROR] [PROCESS_PAGE] Error processing individual table for page {self.page_num}: {table_processing_error}"
                                    )
                                    continue
                            df = pd.concat(df_list)
                            logger.info(
                                f"[DEBUG] [PROCESS_PAGE] Final DataFrame shape: {df.shape if df is not None else 'None'}"
                            )
                            logger.info(
                                f"[DEBUG] [PROCESS_PAGE] Final DataFrame: {df.head() if df is not None else 'None'}"
                            )
                            # Save DataFrame to CSV with error handling
                            if self.temp_path:
                                try:
                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] Saving CSV for page {self.page_num} to: {self.temp_path}"
                                    )
                                    df.to_csv(self.temp_path, index=False)

                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] Successfully saved CSV for page {self.page_num}"
                                    )
                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] CSV file exists: {os.path.exists(self.temp_path)}"
                                    )
                                    logger.info(
                                        f"[DEBUG] [PROCESS_PAGE] CSV file size: {os.path.getsize(self.temp_path) if os.path.exists(self.temp_path) else 'N/A'}"
                                    )
                                except Exception as csv_error:
                                    logger.error(
                                        f"[ERROR] [PROCESS_PAGE] Failed to save CSV for page {self.page_num}: {csv_error}"
                                    )
                                    # Continue processing even if CSV save fails
                            else:
                                logger.warning(
                                    f"[WARNING] [PROCESS_PAGE] No temp_path provided for page {self.page_num}, skipping CSV save"
                                )
                        except Exception as extract_error:
                            logger.error(
                                f"[ERROR] [PROCESS_PAGE] Error extracting tables for page {self.page_num}: {extract_error}"
                            )
                            # Set empty extracted_tables to prevent downstream errors
                            page_data["extracted_tables"] = []
                            df = None
                    else:
                        logger.warning(
                            f"[DEBUG] [PROCESS_PAGE] No tables found in result for page {self.page_num}"
                        )
                        page_data["extracted_tables"] = []
                        df = None
                    page_data["analysis"] = result
                logger.info(
                    f"[DEBUG] [PROCESS_PAGE] Successfully processed page {self.page_num}"
                )
                logger.info(
                    f"[DEBUG] [PROCESS_PAGE] Final DataFrame for page {self.page_num}: {df is not None}"
                )
                return df
            except Exception as e:
                attempt += 1
                logger.error(
                    f"[ERROR] [PROCESS_PAGE] Error processing page {self.page_num}, attempt {attempt}/{self.retries}: {e}"
                )
                if attempt == self.retries:
                    logger.error(
                        f"[ERROR] [PROCESS_PAGE] Failed to process page {self.page_num} after {self.retries} attempts."
                    )
                    return None
                else:
                    logger.error(
                        f"[DEBUG] [PROCESS_PAGE] Retrying page {self.page_num}, attempt {attempt + 1}"
                    )

    def process_pages(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_page = {
                executor.submit(
                    self.process_page, self.page_path, self.page_num, self.retries
                ): self.page_num
                for self.page_num, self.page_path in enumerate(self.images, start=1)
            }
            for future in as_completed(future_to_page):
                self.page_num = future_to_page[future]
                try:
                    page_data = future.result()
                    if page_data is not None:
                        self.combined_data.append(page_data)
                        logger.info(
                            f"[DEBUG] [AZURE_AGENT] Successfully processed page {self.page_num}"
                        )
                    else:
                        logger.warning(
                            f"[WARNING] [AZURE_AGENT] Page {self.page_num} returned None"
                        )
                except Exception as e:
                    logger.error(
                        f"[ERROR] [AZURE_AGENT] Page {self.page_num} failed with error: {e}"
                    )
