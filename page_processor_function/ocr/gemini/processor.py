import json
from google import genai
from google.genai import types
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from utils.pydantic_schema_handler import DynamicModel
from PIL import Image


def prompt(target_schema, xml_output):
    return f"""
            You are an expert bank statement formatter. Your task is to generate the necessary row and column modifications to transform the provided raw table into a final, clean table that adheres to the TARGET SCHEMA. You have access to the original image of the page as a primary reference to accurately identify column boundaries, data types, and formatting discrepancies that may not be fully apparent in the raw XML alone. The series of operations should mandatorily transform the XML to match the provided transaction table in the original image.

            TARGET SCHEMA (or headers) FOR THIS STATEMENT:
            The final, clean table must have exactly these columns in this exact order:

            {target_schema}

            ## PROCESSING WORKFLOW:

            1. Locate the bank transaction table within the document.
            2. Verify the presence of a header row.
            3. Parse and extract all column names from the header row.
            4. Confirm that the column headers in the image correspond to the target schema listed above.
            5. If no header row is detected, return an error message.
            6. Disregard any special characters found within the header labels.

            ### STEPS TO FORMAT BANK STATEMENT:
            1. Analyze the original image thoroughly and understand the data present in the image.
                Things that should be answered:
                - What is the first and the last row of transaction table?
                - How are the columns organised in the transaction table?
                - How are the rows organised in the transaction table?
            2. Now, start analyzing the XML mandatorily with the reference to the understanding of the original image analyzed above.
            3. Now, start suggesting the series of operations that you want to perform on the XML to format the bank statement.
            4. The suggested series of operations should mandatorily transform the XML to match the provided original image of the bank transaction table.
            5. Always refer to the operation definition to understand what each operation does.
            6. Always analyse the results of your set of operations to make sure it matches to the original transaction table provided in the image.
            7. It is mandatory to validate your results before providing the final output.

            ### STEPS TO VALIDATE YOUR OUTPUT:
            1. Use the target schema and original image as reference to analyze whether the series of operations applied on XML transforms it accurately.
            2. If you feel there is any kind of discrepancy in the transformed XML and the provided original image, start the entire process from scratch.

            RULES FOR YOUR ANALYSIS AND OUTPUT:
            Refer to the Original Image: Use the visual layout to resolve ambiguities in RAW TABLE XML—like split cells, misaligned headers, or visually merged columns.

            RAW TABLE XML:
            {xml_output}

            Your output must be a single JSON object structured according to the BankStatementFormattingOutput Pydantic model.

            ### OPERATION DEFINATION:
            1. Tool: regex_replace
                Task: Applies regular expression substitutions to clean or transform text within a table. Each operation defines a regex pattern and a replacement string, which are applied to all string cells in the matrix. Non-matching or non-string values remain unchanged.
                Used for: Cleaning up unwanted characters, formatting inconsistencies, or standardizing values in tabular data extracted from documents.
                Required params:
                    operations: list[dict]
                        regex_replace: list of regex operations with:
                            - regex (str): pattern to search for
                            - replacement (str): string to replace matches with

            2. Tool: delete_rows
                Task: Removes specific rows from a table based on their indices. This operation is used to discard unwanted, empty, or non-informative rows from the matrix.
                Used for: Cleaning tables by eliminating irrelevant or noisy data rows.
                Required params:
                    operations: list[dict]
                        delete_rows: list of operations with:
                            - row_indices (list[int]): indices of rows to be deleted

            3. Tool: map_column
                Task: Assigns or updates header names for specific column indices in a table. Can be scoped to a row range, though the change affects only the header.
                Used for: Renaming or standardizing column headers in a matrix to match a target schema.
                Required params:
                    operations: list[dict]
                        map_column: list of operations with:
                            - header_name (str): name to assign to the column(s)
                            - column_index (list[int]): indices of columns to rename
                            - row_range (dict, optional): defines context range with:
                                - start_row (int)
                                - end_row (int)

            4. Tool: merge_rows
                Task: Combines data from multiple source rows into a single target row in a table. Each column's content from the source rows is merged with the corresponding cell in the target row.
                Used for: Consolidating fragmented row data (e.g., multi-line entries) into a single unified row.
                Required params:
                    operations: list[dict]
                        merge_rows: list of operations with:
                            - source_row_indices (list[int]): rows to merge into the target
                            - target_row_index (int): row that will receive the merged values

            5. Tool: split_cols
                Task: Splits the content of a source column into multiple target columns using a defined logic (typically regex). Only rows within the specified range are processed. Each matched group is mapped to a specific destination column.
                Used for: Breaking down composite column values (e.g., "Type + Amount") into structured, separate columns for cleaner schema alignment.
                Required params:
                    operations: list[dict]
                        split_cols: list of operations with:
                            - source_col_index (int): column to split
                            - num_target_cols (int): expected number of split segments
                            - split_logic (str): regex pattern used to extract values
                            - method (str): currently supports "regex"
                            - row_range (dict, optional): limits processing to:
                                - start_row (int)
                                - end_row (int)
                            - index_created (list[int]): destination column indices for split values
                            - data_mapping (list[str], optional): optional names for mapped values
                            - parameters (list, optional): reserved for custom split options

                Behavior:
                    For each row in the given row_range, the source_col_index is read and matched against the split_logic regex.
                    If the regex matches, each capture group is extracted and inserted into the corresponding column specified in index_created.
                    If the match fails, empty strings are inserted into those positions instead. The operation updates the matrix in place and returns the modified matrix.

            6. Tool: insert_column
                Task: Inserts a new column into a table at a specified position, optionally initializing all its cells with a default value. The new column is also added to the header with a given name.
                Used for: Adding empty or default-filled columns to prepare the table for future data mapping or schema alignment.
                Required params:
                    operations: list[dict]
                        insert_column: list of operations with:
                            - name (str): name of the new column to add to the header
                            - position (int): index at which to insert the new column
                            - default_value (str, optional): initial value for all cells (defaults to empty string if not specified or set to "NaN")

            7. Tool: copy_item
                Task: Copies the value from a specific cell in the matrix and places it into another target cell. Only one cell-to-cell copy is performed per operation.
                Used for: Duplicating data between cells when values need to be reused, propagated, or repositioned across the table.
                Required params:
                    operations: list[dict]
                        copy_item: list of operations with:
                            - from_row (int): source row index
                            - from_col (int): source column index
                            - to_row (int): destination row index
                            - to_col (int): destination column index

            8. Tool: merge_cols
                Task: Merges data from multiple source columns into a single target column for each row, optionally restricted to a row range. The merged values are concatenated and stored in the target column.
                Used for: Combining fragmented data across columns (e.g., first name + last name, or amount + currency) into a single unified column.
                Required params:
                    operations: list[dict]
                        merge_cols: list of operations with:
                            - source_col_indices (list[int]): columns whose values will be merged
                            - target_col_index (int): column where merged value will be placed
                            - row_range (dict, optional): limits processing to:
                                - start_row (int)
                                - end_row (int)

            9. Tool: delete_cols
                Task: Removes specified columns from the matrix. Column indices can be optionally scoped to a row range, though deletion affects the full matrix structure.
                Used for: Dropping irrelevant, empty, or redundant columns to clean and align table data with the target schema.
                Required params:
                    operations: list[dict]
                        delete_cols: list of operations with:
                            - col_indices (list[int]): indices of columns to delete
                            - row_range (dict, optional): defines context range (not used structurally):
                                - start_row (int)
                                - end_row (int)

            ### IMPORTANT:
            - If multiple physical columns are mapped into one logical column, the `merge_cols` operation should be used to combine them. `map_column` is used only to assign headers, not to merge data.
            - Whenever a `merge_cols` operation is used, the residual source columns **must** be deleted using a `delete_cols` operation *after* the merge.
            - Maintain the integrity of the data and the original data as much as possible.

            ## COLUMN TRANSFORMATIONS (Order of Application within the execution cycle):

            1.  delete_cols: Perform initial deletion of physical column indices if they are clearly not present in the target schema or are irrelevant to the target schema (e.g., completely empty columns, indexing columns not part of data).

            2.  split_cols: Split a column into multiple columns based on the information present. Use regex logic. This operation will create new columns at the specified `index_created`.
                ### STEPS TO PERFORM SPLIT_COLS:
                    1. Use regex to match the string that is meant to be extracted for the target columns.
                    2. Utilize the `split_cols` operation to create the new columns at their desired positions (`index_created`).
                    3. After `split_cols`, use `map_column` operations to assign correct headers to these newly created columns if `data_mapping` was not sufficient or if further renaming is needed.
                    4. If the target column already exists in the XML before splitting, you might `merge_cols` the newly created column with the existing column after the split, and then `delete_cols` the source of the split.
                ### NOTE:
                - Use pre-existing column to split the column if it is present in the target schema.
                - Provide row-range to avoid global split and streamline operations.

            3.  insert_column: Add derived or missing columns that were not created by `split_cols`. Specify:
                - position: Index where to insert the new column.
                - name: Column name
                - default_value: Default/fallback if value is not derivable (e.g., for empty 'Debit'/'Credit' columns before population).

            4.  merge_cols: Merge multiple fragmented physical columns into one logical column according to the target schema and image. `merge_cols` concatenates values of source columns into the target column.
                ### NOTE:
                - Remember to follow `merge_cols` with `delete_cols` to remove the source columns.
                - The target column should ideally be at the index where it is present in the target schema.
                - Only merge a row into the one above if:
                    - The row has no date, AND
                    - The row has no money out, money in, or balance values, AND
                    - The row only contains description or partial fragments.
                Otherwise, treat it as a separate transaction.

            5.  map_column: After column structure changes (deletions, splits, insertions, merges), use `map_column` to assign final logical column names from the `TARGET SCHEMA` to their corresponding physical column indices.
                - For each distinct section of transaction data, map physical column indices to logical column names. These mappings refer to the columns *after* any `delete_cols`, `split_cols`, `insert_column`, or `merge_cols` operations have been applied.
                - If a logical column's data spans multiple physical columns (e.g., description combined from two columns after a split), list all relevant physical column indices, and ensure a `merge_cols` operation has already combined them.
                - IMPORTANT: If you encounter any empty or unnamed column in the target schema (like ''), you MUST merge its content (or the content of a relevant adjacent column if it's a visual continuation) with the most relevant details or description column by listing multiple physical column indices for that details column in a `merge_cols` operation. Do not create separate mappings for empty column headers.

            ## ROW TRANSFORMATIONS (Order of Application within the execution cycle):

            1.  split_rows: Split a row that contains multiple transactions (identified visually or through delimiter presence) into separate rows.
            2.  merge_rows: Combine multiple rows that represent a single transaction into one.
                ### NOTE:
                    - The target row should be at an index that makes sense in the overall transaction flow.
                    - The source rows should be deleted using `delete_rows` after merging them into the target row, as `merge_rows` does not delete source rows on its own.
            3.  delete_rows: Remove any rows that are duplicate of other rows, initial headers, summaries, non-transactional rows, metadata related rows, or completely blank.
                ### NOTE:
                    - Prioritize deleting rows that are not part of the *actual transaction data* (e.g., rows without debit or credit values, unless they are clearly continuations of valid transactions to be merged).
                    - Make sure you clean up residual rows *after* performing `merge_rows` and `split_rows` operations to avoid redundancy.
                    - You can delete any detected header rows here, as schema mapping will handle the final headers.
                    - You can also delete the rows like "Balance Brought Forward", "Balance carried forward", "Start Balance", "End Balance", etc. as they do not contribute to the transaction data and are also considered as metadata.

            ## OTHER SUPPORTED ACTIONS (Apply as needed at any point):
            - copy_item: Copy a value from one cell to another.
            - regex_replace: Reserved for rare cases where string cleanup is necessary (can be used any time to clean cell values).

            ### Processing Sequence (Overall Execution Flow of Operation Types):
            Apply transformations in this general order. Remember, individual operations of each type can be called multiple times as needed, and subsequent calls will use indices based on the state of the matrix *after* previous operations.

            1.  Initial delete_cols: Remove columns that are clearly irrelevant or empty at the very beginning.
            2.  Initial delete_rows: Remove rows that are clearly not part of any transaction data (e.g., document headers/footers, summary tables, page numbers) to narrow down the working set.
            3.  Column Refinement:
                a.  `split_cols` (creates new columns)
                b.  `insert_column` (adds empty/default columns not derived from splits)
                c.  `merge_cols` (combines data from multiple columns into one; requires subsequent `delete_cols` for source columns)
                d.  `map_column` (assigns final target schema headers to the processed physical columns)
                e.  Residual `delete_cols`: Delete any remaining columns that are no longer needed after column transformations (e.g., source columns after a merge).
            4.  Row Refinement:
                a.  `split_rows`
                b.  `merge_rows` (requires subsequent `delete_rows` for source rows)
                c.  Residual `delete_rows`: Delete any remaining non-transactional or empty rows that result from row transformations.

            Operation Execution Cycle:
            - Every operation mentioned in output is executed sequentially one at a time.
            - Therefore, the changes in dataframe occurs after each operation is executed.
            - The index mentioned in each operation should be decided based on the changes in the dataframe occurred in the previous operation.

            ### REMEMBER:
            - Use numpy-native logic (str.split, str.extract, str.slice) over complex regex when possible.
            - Design your steps so they are directly usable in a numpy DataFrame transformation pipeline.
            - Prefer clarity, data integrity, and schema alignment over aggressive cleaning.
            - Do not remove signs like +, -, etc from amount columns like deposit, withdrawal, etc.
            - If you face any inconsistency in cell values, then use the same standard for all the rows in that column. All rows should follow the same standard as given in target schema and image provided to you.
            - Before applying any operation, always check the current state of the matrix and the target schema. Also, check that the operation is necessary according to image provided to you.
            - Operations can be applied multiple times throughout the process as needed, with indices adjusted based on the current matrix state.

            ### IMPORTANT INSTRUCTIONS TO TAKE CARE OF IN ORDER TO EXTRACT THE TRANSACTIONS:
            - MAINTAIN the original data exactly as provided in image.
            - Do not make assumptions about missing information or formatting.
            - Any row with either a debit or a credit value is always considered as a transaction.
            - Any row without any debit or credit value is either a part of other transaction (lying above or below) and has to be merged in actual transaction or it can be a meta-data transaction or not a transaction at all.
            - Data other than transactional data should be dropped to maintain the integrity and originality of the dataframe as we are not interested in that data and we are only working on actual transactional data.
            - Also note that i have shifted from working with dataframe to working with 2D matrices using numpy. Therefore i want you to provide me replacement for numpy specific operations that can be used in numpy 2D matrix where ever necessary if any.
            - If you are unable to extract any table, you can delete all rows to return an empty matrix.

            ### FLOW OF EXECUTION (High-Level Initial Filtering):
            - First, you should perform an initial deletion of columns and rows that are clearly unnecessary or non-transactional (e.g., document headers, footers, irrelevant sections, completely empty columns) to narrow down the scope to potential transaction data. This is a primary cleanup before detailed transformations.
        """


class GeminiAgent:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GeminiApiKey")
        )
        self.model = "gemini-2.5-pro"

    def call_gemini(self, xml_output, image, headers):

        prompt_text = prompt(headers, str(xml_output))

        logger.info(f"Prompt Text: {prompt_text}")
        image = Image.open(image)

        content = [prompt_text, image]

        thoughts = ""
        answer = ""

        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DynamicModel,
                temperature=1,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
            contents=content,
        )

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            elif part.thought:
                if not thoughts:
                    logger.info("Thoughts summary:")
                logger.info(part.text)
                thoughts += part.text
            else:
                if not answer:
                    logger.info("Answer:")
                logger.info(part.text)
                answer += part.text

        json_response = json.loads(response.text)["operations"]
        return json_response
