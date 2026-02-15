"""
Pydantic models for the CrewAI Flow state.
Tracks data as it moves through the pipeline: upload -> schema -> clean -> DB -> stats -> report.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """Schema information for a single column."""
    name: str
    dtype: str
    null_count: int = 0
    unique_count: int = 0
    sample_values: list[str] = Field(default_factory=list)


class SchemaInfo(BaseModel):
    """Full schema summary of the uploaded CSV."""
    filename: str
    row_count: int
    column_count: int
    columns: list[ColumnInfo] = Field(default_factory=list)
    summary_stats: str = ""  # Textual summary from the agent


class CleaningReport(BaseModel):
    """Report of all transformations applied during cleaning."""
    rows_before: int = 0
    rows_after: int = 0
    columns_dropped: list[str] = Field(default_factory=list)
    nulls_filled: dict[str, str] = Field(default_factory=dict)  # col -> method
    duplicates_removed: int = 0
    outliers_removed: int = 0
    type_conversions: dict[str, str] = Field(default_factory=dict)  # col -> new_type
    notes: str = ""


class TableSchema(BaseModel):
    """Schema for a single table in the relational design."""
    table_name: str
    columns: dict[str, str] = Field(default_factory=dict)  # col_name -> SQL type
    primary_key: str = ""
    foreign_keys: dict[str, str] = Field(default_factory=dict)  # col -> ref table.col
    row_count: int = 0


class DBSchema(BaseModel):
    """Full relational schema designed by the DB Architect agent."""
    db_name: str = ""
    tables: list[TableSchema] = Field(default_factory=list)
    design_rationale: str = ""  # Agent's explanation of the schema design


class TestResult(BaseModel):
    """Result of a single statistical test."""
    test_name: str
    columns_tested: list[str] = Field(default_factory=list)
    statistic: float = 0.0
    p_value: float = 0.0
    significant: bool = False
    interpretation: str = ""


class ChartInfo(BaseModel):
    """Metadata for a generated chart."""
    chart_type: str
    title: str
    file_path: str
    description: str = ""


class AnalysisState(BaseModel):
    """
    Master state passed through the CrewAI Flow.
    Each agent step populates its relevant section.
    """
    # Input
    csv_path: str = ""
    original_filename: str = ""

    # Schema Analyst output
    schema_info: Optional[SchemaInfo] = None

    # Data Cleaner output
    cleaned_csv_path: str = ""
    cleaning_report: Optional[CleaningReport] = None

    # DB Architect output
    db_schema: Optional[DBSchema] = None

    # Statistical Analyst output
    test_results: list[TestResult] = Field(default_factory=list)
    data_classification: str = ""  # e.g. "mostly numeric", "mixed", "categorical"

    # Report Generator output
    charts: list[ChartInfo] = Field(default_factory=list)
    report_path: str = ""
    report_content: str = ""
