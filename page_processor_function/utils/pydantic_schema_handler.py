import enum
from typing import List
from pydantic import BaseModel


class SchemaType(enum.Enum):
    FINANCIAL = "financial"
    GENERIC = "generic"


class TransactionType(enum.Enum):
    TRANSACTION = "transaction"
    SUMMARY = "summary"


class GenericTransaction(BaseModel):
    type: TransactionType
    date: str
    payment_type: str
    description: str
    amount_column_1: float
    amount_column_2: float
    balance: float
    step_by_step_calculation: str


class FinancialTransaction(BaseModel):
    type: TransactionType
    date: str
    payment_type: str
    description: str
    paid_out: float
    paid_in: float
    balance: float
    step_by_step_calculation: str


class StatementPeriod(BaseModel):
    start_date: str
    end_date: str


class GenericColumnChecksums(BaseModel):
    amount_column_1: float
    amount_column_2: float


class FinancialColumnChecksums(BaseModel):
    paid_out: float
    paid_in: float


class VerificationSummary(BaseModel):
    record_count: int
    financial_column_checksums: FinancialColumnChecksums
    generic_column_checksums: GenericColumnChecksums
    balance_calculation_check: str
    unparsed_lines: List[str]


class BankStatement(BaseModel):
    statement_period: StatementPeriod
    schema_type: SchemaType
    column_mapping_method: str
    financial_transactions: List[FinancialTransaction]
    generic_transactions: List[GenericTransaction]
    verification_summary: VerificationSummary
    is_generic_schema_applicable: bool
    is_consistent: bool
