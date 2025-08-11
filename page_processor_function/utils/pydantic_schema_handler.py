from pydantic import BaseModel
from typing import Optional, List, Literal
from enum import Enum


class CSV_Row(BaseModel):
    rows: List[str]


class OperationType(str, Enum):
    SPLIT_COLS = "split_cols"
    MERGE_COLS = "merge_cols"
    DELETE_COLS = "delete_cols"
    SPLIT_ROWS = "split_rows"
    MERGE_ROWS = "merge_rows"
    DELETE_ROWS = "delete_rows"
    MAP_COLUMN = "map_column"
    REGEX_REPLACE = "regex_replace"
    INSERT_COLUMN = "insert_column"
    COPY_ITEM = "copy_item"


class BaseMapping(BaseModel):
    headers: str  # Logical column name
    column_indices: List[int]  # Source physical columns contributing to this header


class RowRange(BaseModel):
    start_row: int
    end_row: int


class SplitCols(BaseModel):
    row_range: RowRange
    source_col_index: int
    num_target_cols: int
    split_logic: str
    method: Literal["regex"]  # This locks method to only "regex"
    parameters: List[int]
    data_mapping: List[str]
    index_created: List[int]


class MergeCols(BaseModel):
    row_range: RowRange
    source_col_indices: List[int]
    target_col_index: int


class DeleteCols(BaseModel):
    row_range: RowRange
    col_indices: List[int]


class SplitRows(BaseModel):
    source_row_index: int
    num_target_rows: int
    split_logic: str
    method: str
    pattern: str
    data_mapping_to_columns: List[BaseMapping]


class MergeRows(BaseModel):
    source_row_indices: List[int]
    target_row_index: int


class DeleteRows(BaseModel):
    row_indices: List[int]


class MapColumn(BaseModel):
    row_range: RowRange
    header_name: str
    column_index: List[int]


class RegexReplace(BaseModel):
    row_range: RowRange
    regex: str
    replacement: str


class InsertColumn(BaseModel):
    row_range: RowRange
    position: int
    name: str
    default_value: str


class CopyItem(BaseModel):
    from_row: int
    from_col: int
    to_row: int
    to_col: int


class OperationAction(BaseModel):
    split_cols: List[SplitCols] = []
    map_column: List[MapColumn] = []
    merge_cols: List[MergeCols] = []
    split_rows: List[SplitRows] = []
    merge_rows: List[MergeRows] = []
    regex_replace: List[RegexReplace] = []
    insert_column: List[InsertColumn] = []
    copy_item: List[CopyItem] = []
    delete_rows: List[DeleteRows] = []
    delete_cols: List[DeleteCols] = []


class Steps(BaseModel):
    step_number: int
    step_description: str
    step_explanation: str
    step_output: str


class Shape(BaseModel):
    num_of_rows: int
    num_of_columns: int


class Operation(BaseModel):
    step_input_df_shape: Shape
    operation_type: OperationType
    operation: OperationAction
    step_by_step_explanation: List[Steps]
    step_output_df_shape: Shape


class DynamicModel(BaseModel):
    operations: List[Operation]
    headers: List[str]
