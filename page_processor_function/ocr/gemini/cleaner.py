import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def sanitize_regex_pattern(pattern):
    """
    Sanitize and validate regex patterns to prevent syntax errors.

    Args:
        pattern (str): Raw regex pattern

    Returns:
        str: Sanitized regex pattern
    """
    try:
        # Test if pattern compiles correctly
        re.compile(pattern)
        return pattern
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        # Common fixes for invalid patterns
        sanitized = pattern

        # Fix unterminated character classes like [J\]
        if "[" in sanitized and "]" in sanitized:
            # Find problematic character classes
            char_class_pattern = r"\[([^\]]*)\]"
            matches = re.finditer(char_class_pattern, sanitized)

            for match in matches:
                char_class = match.group(0)
                char_class_content = match.group(1)

                # If character class contains unescaped special chars, fix it
                if "\\]" in char_class_content:
                    # Convert [J\] to literal J]
                    fixed_content = char_class_content.replace("\\]", "]")
                    # Remove the brackets to make it literal
                    sanitized = sanitized.replace(char_class, fixed_content)
                    logger.info(
                        f"Fixed character class: {char_class} -> {fixed_content}"
                    )

        # Escape common problematic characters if they're not already escaped
        problematic_chars = [".", "+", "*", "?", "^", "$", "(", ")", "{", "}", "|"]
        for char in problematic_chars:
            if char in sanitized and f"\\{char}" not in sanitized:
                # Only escape if not already escaped and not in a valid context
                sanitized = sanitized.replace(char, f"\\{char}")

        # Test the sanitized pattern
        try:
            re.compile(sanitized)
            logger.info(f"Successfully sanitized pattern: '{pattern}' -> '{sanitized}'")
            return sanitized
        except re.error:
            # If still invalid, treat as literal string
            escaped_pattern = re.escape(pattern)
            logger.warning(
                f"Using literal match for pattern: '{pattern}' -> '{escaped_pattern}'"
            )
            return escaped_pattern


def handle_regex_replace(operations, matrix):
    """
    Apply regex find-and-replace operations to all cells in the matrix with robust error handling.

    This function iterates through each cell in the matrix and applies regex patterns
    to clean data (e.g., removing currency symbols, formatting numbers, etc.)

    Args:
        operations (list): List containing operation dictionary with 'regex_replace' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Modified matrix with regex replacements applied

    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_regex_replace ===")
    logger.info(f"operations: {operations}")

    # Performing operations
    try:
        rows_in_matrix = len(matrix)
        columns_in_matrix = len(matrix[0]) if len(matrix) > 0 else 0

        if rows_in_matrix == 0 or columns_in_matrix == 0:
            logger.warning("Empty matrix provided to regex_replace")
            return matrix

        operation_dict = operations[0]
        regex_replace_ops = operation_dict["regex_replace"]

        # Process each regex replacement operation
        for regex_replace in regex_replace_ops:
            raw_regex = regex_replace["regex"]
            replacement = regex_replace["replacement"]

            # Get row range if specified
            row_range = regex_replace.get("row_range", {})
            start_row = row_range.get("start_row", 0)
            end_row = row_range.get("end_row", rows_in_matrix)

            # Validate row range
            start_row = max(0, min(start_row, rows_in_matrix - 1))
            end_row = max(start_row + 1, min(end_row, rows_in_matrix))

            logger.info(
                f"Processing regex replacement from row {start_row} to {end_row}"
            )
            logger.info(f"Raw regex: '{raw_regex}' -> Replacement: '{replacement}'")

            # Sanitize the regex pattern
            sanitized_regex = sanitize_regex_pattern(raw_regex)

            # Compile pattern once for efficiency
            try:
                compiled_pattern = re.compile(sanitized_regex)
            except re.error as e:
                logger.error(
                    f"Failed to compile regex pattern '{sanitized_regex}': {e}"
                )
                # Skip this operation if pattern still fails
                continue

            replacements_made = 0

            # Apply regex to each cell in the specified range
            for r in range(start_row, end_row):
                for c in range(columns_in_matrix):
                    val = matrix[r, c]
                    # Only process non-empty string values
                    if isinstance(val, str) and val:
                        try:
                            original_val = matrix[r, c]
                            updated_val = compiled_pattern.sub(
                                replacement, original_val
                            )

                            if original_val != updated_val:
                                matrix[r, c] = updated_val
                                replacements_made += 1
                                logger.info(
                                    f"[r={r}, c={c}] '{original_val}' → '{updated_val}' using pattern '{sanitized_regex}'"
                                )
                        except Exception as cell_error:
                            logger.warning(
                                f"Error processing cell [{r},{c}]: {cell_error}"
                            )
                            continue

            logger.info(
                f"Made {replacements_made} replacements with pattern '{sanitized_regex}'"
            )

        return matrix

    except Exception as e:
        logger.error(f"Error in handle_regex_replace: {e}")
        # Return original matrix instead of failing completely
        return matrix


def handle_delete_rows(operations, matrix):
    """
    Delete specified rows from the matrix.

    Removes entire rows based on their indices. Useful for eliminating header rows,
    footer summaries, or invalid data rows.

    Args:
        operations (list): List containing operation dictionary with 'delete_rows' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Matrix with specified rows removed

    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_delete_rows ===")
    logger.info(f"operations: {operations}")

    # Performing operations
    try:
        operation_dict = operations[0]
        delete_rows_ops = operation_dict["delete_rows"]

        for delete_rows in delete_rows_ops:
            row_indices = delete_rows.get("row_indices", [])
            # Validate indices are within bounds
            valid_indices = [i for i in row_indices if 0 <= i < matrix.shape[0]]
            if valid_indices:
                logger.info(f"Deleting rows {valid_indices}\n")
                matrix = np.delete(matrix, valid_indices, 0)
            else:
                logger.warning("No valid row indices to delete")

        return matrix

    except Exception as e:
        raise Exception(f"Error in handle_delete_rows: {e}")


def handle_map_column(operations, matrix, header):
    """
    Map column headers to specific positions in the header array.

    Updates the header row to assign proper column names to specific positions.
    This is useful for standardizing column names across different document formats.

    Args:
        operations (list): List containing operation dictionary with 'map_column' key
        matrix (numpy.ndarray): 2D array representing the data table
        header (numpy.ndarray): 1D array representing column headers

    Returns:
        tuple: (matrix, updated_header) - Matrix unchanged, header with new mappings

    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_map_column ===")
    logger.info(f"operations: {operations}")

    # Performing operations
    try:
        operation_dict = operations[0]
        map_column_ops = operation_dict["map_column"]

        for map_column_op in map_column_ops:
            header_name = map_column_op["header_name"]
            col_indices = map_column_op["column_index"]
            row_range = map_column_op.get("row_range", {})
            start_row = row_range.get("start_row", 0)
            end_row = row_range.get("end_row", len(header))

            logger.info(
                f"Mapping {header_name} to index {col_indices} within the range of {start_row} to {end_row}\n"
            )

            # Apply header name to specified column indices
            for col_index in col_indices:
                if col_index < len(header):
                    header[col_index] = header_name
                else:
                    header = np.insert(header, col_index, header_name)

            logger.info(f"header after mapping: \n {header}")

        logger.info(f"ending state of header: \n {header}")
        return matrix, header

    except Exception as e:
        raise Exception(f"Error in handle_map_column: {e}")


def handle_merge_rows(operations, matrix):
    """
    Merge multiple rows by concatenating their cell values.

    Combines data from source rows into a target row. Useful when data is split
    across multiple rows due to formatting issues in the source document.

    Args:
        operations (list): List containing operation dictionary with 'merge_rows' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Matrix with rows merged (source rows remain but target updated)


    Note: Values are concatenated with space separator. Order depends on row indices.
    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_merge_rows ===")
    logger.info(f"operations: {operations}")

    # Check if matrix is empty
    if matrix.size == 0 or matrix.shape[0] == 0:
        logger.info("Matrix is empty, skipping merge_rows operation")
        return matrix

    operation_dict = operations[0]
    merge_ops = operation_dict["merge_rows"]

    # Performing operations
    try:
        rows_in_matrix = len(matrix)
        columns_in_matrix = len(matrix[0])

        for merge_op in merge_ops:
            row_range = merge_op.get("row_range", {})
            start_row = row_range.get("start_row", 0)
            end_row = row_range.get("end_row", len(matrix))
            if end_row < 0:
                end_row = len(matrix)
            source_row_indices = merge_op.get("source_row_indices", [])
            target_row_index = merge_op.get("target_row_index")

            logger.info(
                f"Merging {source_row_indices} to {target_row_index} within the range of {start_row} to {end_row}\n"
            )

            # Process each source row within the specified range
            for rows in source_row_indices:
                if start_row <= rows <= end_row:
                    if not (0 <= target_row_index < matrix.shape[0]):
                        raise ValueError("Invalid target row index")

                    # Merge each column of the source and target rows
                    for c in range(columns_in_matrix):
                        logger.info(
                            f"merging row {rows} and {target_row_index} at column {c}"
                        )
                        logger.info(f"source row value: {matrix[rows, c]}")
                        logger.info(f"target row value: {matrix[target_row_index, c]}")

                        # Concatenate values based on row order
                        if rows > target_row_index:
                            merged_val = (
                                f"{matrix[target_row_index, c]} {matrix[rows, c]}"
                            )
                        else:
                            merged_val = (
                                f"{matrix[rows, c]} {matrix[target_row_index, c]}"
                            )

                        logger.info(
                            f"merging row {rows} and {target_row_index} at column {c}\n Final element value: {merged_val}"
                        )
                        matrix[target_row_index, c] = merged_val

        return matrix

    except Exception as e:
        raise Exception(f"Error in handle_merge_rows: {e}")


def handle_split_cols(operations, matrix):
    """
    Split a single column into multiple columns using regex patterns.

    Uses regex with capture groups to extract different parts of a cell value
    into separate columns. Commonly used for splitting date/time strings or
    combined transaction descriptions.

    Args:
        operations (list): List containing operation dictionary with 'split_cols' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Matrix with source column data split into target columns

    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_split_cols ===")
    logger.info(f"operations: {operations}")

    try:
        operation_dict = operations[0]
        split_cols_ops = operation_dict["split_cols"]

        for split_cols in split_cols_ops:
            row_range = split_cols.get("row_range", {})
            start_row = row_range.get("start_row", 0)
            end_row = row_range.get("end_row", len(matrix))
            if end_row < 0:
                end_row = len(matrix)
            src_idx = split_cols.get("source_col_index")
            split_logic = split_cols.get("split_logic", "")
            method = split_cols.get("method", "regex")
            num_target_cols = split_cols.get("num_target_cols")
            params = split_cols.get("parameters", [])
            data_mapping = split_cols.get("data_mapping", [])
            index_created = split_cols.get("index_created", [])

            logger.info(
                f"Splitting column {src_idx} using method {method} with logic '{split_logic}' "
                f"and data mapping to columns {data_mapping} for row range {start_row} to {end_row}. "
                f"Parameters: {params}"
            )

            # Compile regex pattern for efficiency
            pattern = re.compile(split_logic)

            # Process each row in the specified range
            for row_idx in range(start_row, end_row):
                cell = matrix[row_idx, src_idx]
                match = pattern.match(cell)

                if match:
                    groups = list(match.groups())
                else:
                    # Fill with empty strings if no match
                    groups = [""] * num_target_cols

                # Assign matched groups to target columns
                for i, destination_id in enumerate(index_created):
                    if i < len(groups):
                        logger.info(
                            f"row {row_idx} original: '{cell}' -> groups: {groups}"
                        )
                        matrix[row_idx, destination_id] = groups[i]

        logger.info(f"matrix post split_cols: \n {matrix}")
        return matrix

    except Exception as e:
        raise Exception(f"Error in handle_split_cols: {e}")


def handle_insert_column(operations, matrix, header):
    """
    Insert new columns at specified positions with default values.

    Adds new columns to the matrix and updates the header accordingly.
    Useful for adding calculated fields or standardizing table structure.

    Args:
        operations (list): List containing operation dictionary with 'insert_column' key
        matrix (numpy.ndarray): 2D array representing the data table
        header (numpy.ndarray): 1D array representing column headers

    Returns:
        tuple: (updated_matrix, updated_header) with new columns inserted


    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_insert_column ===")
    logger.info(f"operations: {operations}")

    # Performing operations
    try:
        operation_dict = operations[0]
        insert_column_ops = operation_dict["insert_column"]

        for insert_column in insert_column_ops:
            column_name = insert_column["name"]
            index = insert_column["position"]
            # Ensure index is within valid range
            index = min(index, matrix.shape[1])
            default_value = insert_column.get("default_value", "")

            # Handle special case for NaN values
            if default_value.lower() == "nan":
                default_value = ""

            logger.info(
                f"Inserting column {column_name} at index {index} using default value '{default_value}'\n"
            )

            # Create a column filled with default_value
            default_col = np.full((matrix.shape[0],), default_value, dtype=object)
            logger.info(f"default_col: {default_col}")

            # Insert column into matrix and header
            matrix = np.insert(matrix, index, default_col, axis=1)
            header = np.insert(header, index, column_name)
            logger.info(f"header after inserting column {header} at index {index}")

        return matrix, header

    except Exception as e:
        raise Exception(f"Error in handle_insert_column: {e}")


def handle_copy_item(operations, matrix):
    """
    Copy individual cell values from one position to another.

    Copies the value from a source cell to a target cell. Useful for filling
    missing data or duplicating values across the table.

    Args:
        operations (list): List containing operation dictionary with 'copy_item' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Matrix with copied values


    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_copy_item ===")
    logger.info(f"operations: {operations}")

    # Performing operations
    try:
        operation_dict = operations[0]
        copy_item_ops = operation_dict["copy_item"]

        for copy_item in copy_item_ops:
            from_row = copy_item["from_row"]
            from_col = copy_item["from_col"]
            to_row = copy_item["to_row"]
            to_col = copy_item["to_col"]

            logger.info(
                f"Copying item from row {from_row} and column {from_col} to row {to_row} and column {to_col}\n"
            )
            logger.info(
                f"Copied {matrix[from_row, from_col]} to {matrix[to_row, to_col]}"
            )

            # Validate all indices are within bounds
            if not (
                0 <= from_row < matrix.shape[0]
                and 0 <= from_col < matrix.shape[1]
                and 0 <= to_row < matrix.shape[0]
                and 0 <= to_col < matrix.shape[1]
            ):
                raise ValueError("Copy indices out of range")

            # Perform the copy operation
            matrix[to_row, to_col] = matrix[from_row, from_col]

        return matrix

    except Exception as e:
        raise Exception(f"Error in handle_copy_item: {e}")


def handle_merge_cols(operations, matrix):
    """
    Merge multiple columns by concatenating their values row-wise.

    Combines values from source columns into a target column for each row.
    Useful for creating combined fields like "Full Name" from separate
    first and last name columns.

    Args:
        operations (list): List containing operation dictionary with 'merge_cols' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Matrix with merged column values


    Note: Values are concatenated with space separator. Order based on column indices.
    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info(f"=== Starting handle_merge_cols ===")
    logger.info(f"operations: {operations}")

    # Check if matrix is empty
    if matrix.size == 0 or matrix.shape[0] == 0:
        logger.info("Matrix is empty, skipping merge_cols operation")
        return matrix

    # Performing operations
    try:
        operation_dict = operations[0]
        merge_cols_ops = operation_dict["merge_cols"]

        for merge_cols in merge_cols_ops:
            row_range = merge_cols.get("row_range", {})
            target_col_index = merge_cols.get("target_col_index")
            # Exclude target column from source list and remove duplicates
            source_col_indices = [
                i
                for i in merge_cols.get("source_col_indices", [])
                if i != target_col_index
            ]
            source_col_indices = list(set(source_col_indices))
            start_row = row_range.get("start_row", 0)
            end_row = row_range.get("end_row", len(matrix))
            if end_row < 0:
                end_row = len(matrix)

            logger.info(
                f"Copying from index {source_col_indices} to {target_col_index} within the range of {start_row} to {end_row}\n"
            )

        # Process each row in the specified range
        for r in range(start_row, end_row):
            for source_col_index in source_col_indices:
                logger.info(
                    f"merging column {source_col_index} and {target_col_index} at row {r}"
                )
                logger.info(f"source column value: {matrix[r, source_col_index]}")
                logger.info(f"target column value: {matrix[r, target_col_index]}")

                # Concatenate values based on column order
                if source_col_index < target_col_index:
                    merged_val = (
                        f"{matrix[r, source_col_index]} {matrix[r, target_col_index]}"
                    )
                else:
                    merged_val = (
                        f"{matrix[r, target_col_index]} {matrix[r, source_col_index]}"
                    )

                logger.info(
                    f"merging column {source_col_index} and {target_col_index} at row {r}\n Final row: {merged_val}"
                )
                matrix[r, target_col_index] = merged_val
            logger.info(f"row {r} after merging: {matrix[r]}")

        return matrix

    except Exception as e:
        raise Exception(f"Error in handle_merge_cols: {e}")


def handle_delete_cols(operations, matrix):
    """
    Delete specified columns from the matrix.

    Removes entire columns based on their indices. Useful for eliminating
    unwanted or empty columns from the extracted data.

    Args:
        operations (list): List containing operation dictionary with 'delete_cols' key
        matrix (numpy.ndarray): 2D array representing the data table

    Returns:
        numpy.ndarray: Matrix with specified columns removed


    Note: Columns are deleted in reverse order to avoid index shifting issues.
    """
    logger.info(f"starting state of dataframe: \n {matrix}")
    logger.info("=== Starting handle_delete_cols ===")
    logger.info(f"operations: {operations}")

    # Performing operations
    try:
        operation_dict = operations[0]
        delete_cols_ops = operation_dict["delete_cols"]
        logger.info(f"delete_cols_ops: {delete_cols_ops}")

        for delete_cols in delete_cols_ops:
            row_range = delete_cols.get("row_range", {})
            column_indices = delete_cols.get("col_indices", [])
            start_row = row_range.get("start_row", 0)
            end_row = row_range.get("end_row", len(matrix))
            if end_row < 0:
                end_row = len(matrix)

            logger.info(
                f"Deleting column at index {column_indices} within the range of {start_row} to {end_row}\n"
            )

            # Delete columns in reverse order to avoid index shifting
            for col_index in sorted(column_indices, reverse=True):
                if col_index < matrix.shape[1]:
                    logger.info(f"deleting column {col_index}")
                    logger.info(f"column value: {matrix[0, col_index]}")
                    matrix = np.delete(matrix, col_index, 1)
                    logger.info(f"column {col_index} deleted")

        return matrix

    except Exception as e:
        raise Exception(f"Error in handle_delete_cols: {e}")


# Operation handler mapping - maps operation types to their handler functions
OPERATION_HANDLERS = {
    "regex_replace": handle_regex_replace,
    "delete_rows": handle_delete_rows,
    "map_column": handle_map_column,
    "merge_rows": handle_merge_rows,
    "split_cols": handle_split_cols,
    "insert_column": handle_insert_column,
    "copy_item": handle_copy_item,
    "merge_cols": handle_merge_cols,
    "delete_cols": handle_delete_cols,
}


def process_gemini_response(json_response, temp_path, extracted_data_path):
    """
    Main processing function that applies a sequence of operations to clean tabular data.

    This function orchestrates the entire data cleaning process by:
    1. Loading CSV data into a numpy matrix
    2. Processing each operation in sequence
    3. Saving the cleaned data back to CSV

    Args:
        json_response (dict|list): Gemini AI response containing operations to perform
        temp_path (str): Path to input CSV file with raw extracted data
        extracted_data_path (str): Path where cleaned CSV data should be saved

    Returns:
        None: Results are saved to extracted_data_path

    Raises:
        Exception: If no transactions found after processing or operation failures

    """

    logger.info("=== Starting GeminiResponseHandler execution ===")

    # Load CSV data into numpy matrix
    df = pd.read_csv(temp_path)
    matrix = df.to_numpy()

    # Check if matrix is empty
    if matrix.size == 0 or matrix.shape[0] == 0:
        logger.info("Warning: Empty matrix loaded from CSV")
        # Create an empty DataFrame and save it
        empty_df = pd.DataFrame()
        empty_df.to_csv(extracted_data_path, index=False)
        logger.info("Saved empty CSV file")
        return

    header = matrix[0]  # First row contains headers
    logger.info(f"matrix (2D array) obtained from dataframe: \n {matrix}")
    logger.info(f"matrix shape: {matrix.shape}")

    # Extract operations from response (handle both dict and list formats)
    operations = (
        json_response.get("operations", [])
        if isinstance(json_response, dict)
        else json_response
    )

    # Replace NaN values with empty strings for consistent processing
    matrix = np.where(
        matrix == matrix,  # checks for non-NaN (nan != nan)
        matrix,  # keep original if not NaN
        "",  # replace if NaN
    )
    logger.info(f"matrix after replacing np.nan with empty string: \n {matrix}")

    # Process each operation sequentially
    for op in operations:
        op_type = op.get("operation_type")
        op_details = op.get("operation", {})
        logger.info(f"Starting operation: {op_type} with details: {op_details}")

        # Get appropriate handler for this operation type
        handler = OPERATION_HANDLERS.get(op_type)
        if handler:
            try:
                logger.info(f"Applying operation: {op_type}")

                # Special handling for operations that modify headers
                if op_type == "map_column" or op_type == "insert_column":
                    matrix, header = handler([op_details], matrix, header=header)
                else:
                    matrix = handler([op_details], matrix)

                logger.info(f"matrix post {op_type}: \n {matrix}")
                logger.info(f"Completed operation: {op_type}")
                logger.info(f"Post operation shape of matrix: {matrix.shape}")

            except Exception as e:
                raise Exception(f"Operation {op_type} failed: {e}")

        else:
            logger.warning(f"No handler found for operation type: {op_type}")

    # Validate final result and save to CSV
    if len(matrix) == 0:
        logger.error("No transactions found in the dataframe")
        return

    # Create DataFrame with proper headers if possible
    if len(header) == len(matrix[0]):
        df = pd.DataFrame(matrix, columns=header)
        df.to_csv(extracted_data_path, index=False)
    else:
        # Fallback: save without custom headers if mismatch
        df = pd.DataFrame(matrix)
        df.to_csv(extracted_data_path, index=False)

    logger.info(f"Final matrix:\n {matrix}")
