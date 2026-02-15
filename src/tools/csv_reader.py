"""
CSVReaderTool -- Reads and analyzes CSV files.
Extracts schema information: column names, data types, null counts, shape, and summary stats.
"""

import json
import chardet
import pandas as pd
from crewai.tools import tool


def _detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, "rb") as f:
        raw = f.read(10000)  # Read first 10KB for detection
    result = chardet.detect(raw)
    return result.get("encoding", "utf-8") or "utf-8"


@tool("CSV Reader")
def csv_reader_tool(file_path: str) -> str:
    """
    Read a CSV file and return its schema information.
    Returns a JSON string with column names, data types, shape, null counts,
    unique counts, sample values, and basic summary statistics.

    Args:
        file_path: Absolute path to the CSV file to analyze.
    """
    try:
        # Detect encoding
        encoding = _detect_encoding(file_path)

        # Read CSV with fallback encoding
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, LookupError):
            df = pd.read_csv(file_path, encoding="utf-8", errors="replace", low_memory=False)

        # Guard: empty CSV
        if df.empty or len(df.columns) == 0:
            return json.dumps({"error": "CSV file is empty or has no columns."})

        # Build column info
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
            }
            columns.append(col_info)

        # Summary statistics for numeric columns
        numeric_summary = ""
        if not df.select_dtypes(include=["number"]).empty:
            numeric_summary = df.describe().to_string()

        # Categorical summary
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_summary = ""
        if categorical_cols:
            cat_lines = []
            for col in categorical_cols:
                top_values = {str(k): int(v) for k, v in df[col].value_counts().head(5).items()}
                cat_lines.append(f"  {col}: {top_values}")
            cat_summary = "Categorical distributions:\n" + "\n".join(cat_lines)

        result = {
            "filename": file_path.split("\\")[-1].split("/")[-1],
            "rows": len(df),
            "columns_count": len(df.columns),
            "columns": columns,
            "numeric_summary": numeric_summary,
            "categorical_summary": cat_summary,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
