"""
DataCleanerTool -- Cleans and preprocesses CSV data.
Handles nulls, duplicates, type conversions, and outlier removal.
"""

import json
import pandas as pd
import numpy as np
from crewai.tools import tool


@tool("Data Cleaner")
def data_cleaner_tool(file_path: str, instructions: str) -> str:
    """
    Clean a CSV file based on agent instructions.
    Performs: duplicate removal, null handling, type fixes, and outlier removal.
    Saves the cleaned data to a new file and returns a cleaning report.

    Args:
        file_path: Absolute path to the CSV file to clean.
        instructions: JSON string with cleaning instructions from the agent.
                      Example: {"drop_duplicates": true, "fill_nulls": {"age": "median", "name": "drop"},
                                "convert_types": {"price": "float"}, "remove_outliers": ["salary"]}
    """
    try:
        df = pd.read_csv(file_path)
        report = {
            "rows_before": len(df),
            "columns_dropped": [],
            "nulls_filled": {},
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "type_conversions": {},
            "notes": "",
        }

        # Parse instructions
        try:
            config = json.loads(instructions)
        except (json.JSONDecodeError, TypeError):
            config = {}

        # 1. Drop duplicates
        if config.get("drop_duplicates", True):
            before = len(df)
            df = df.drop_duplicates()
            report["duplicates_removed"] = before - len(df)

        # 2. Handle nulls
        null_config = config.get("fill_nulls", {})
        for col, method in null_config.items():
            if col not in df.columns:
                continue
            if method == "drop":
                df = df.dropna(subset=[col])
                report["nulls_filled"][col] = "dropped rows"
            elif method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].mean()
                if pd.notna(fill_val):
                    df[col] = df[col].fillna(fill_val)
                    report["nulls_filled"][col] = "mean"
            elif method == "median" and pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                if pd.notna(fill_val):
                    df[col] = df[col].fillna(fill_val)
                    report["nulls_filled"][col] = "median"
            elif method == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
                report["nulls_filled"][col] = "mode"
            elif method == "zero":
                df[col] = df[col].fillna(0)
                report["nulls_filled"][col] = "zero"
            else:
                # Treat as a literal fill value
                df[col] = df[col].fillna(method)
                report["nulls_filled"][col] = f"literal: {method}"

        # Auto-handle remaining nulls: drop rows with >50% nulls, fill numeric with median, categorical with mode
        if config.get("auto_clean_remaining", True):
            # Drop rows that are mostly empty
            threshold = len(df.columns) * 0.5
            before = len(df)
            df = df.dropna(thresh=int(threshold))
            dropped = before - len(df)
            if dropped > 0:
                report["notes"] += f"Dropped {dropped} rows with >50% null values. "

            # Fill remaining numeric nulls with median
            for col in df.select_dtypes(include=["number"]).columns:
                if df[col].isnull().any() and col not in null_config:
                    fill_val = df[col].median()
                    if pd.notna(fill_val):
                        df[col] = df[col].fillna(fill_val)
                        report["nulls_filled"][col] = "auto: median"

            # Fill remaining categorical nulls with mode
            for col in df.select_dtypes(include=["object", "category"]).columns:
                if df[col].isnull().any() and col not in null_config:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
                    df[col] = df[col].fillna(mode_val)
                    report["nulls_filled"][col] = "auto: mode"

        # 3. Type conversions
        type_config = config.get("convert_types", {})
        for col, target_type in type_config.items():
            if col not in df.columns:
                continue
            try:
                if target_type == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif target_type == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif target_type == "str":
                    df[col] = df[col].astype(str)
                elif target_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif target_type == "category":
                    df[col] = df[col].astype("category")
                report["type_conversions"][col] = target_type
            except Exception as e:
                report["notes"] += f"Failed to convert {col} to {target_type}: {e}. "

        # 4. Remove outliers (IQR method)
        outlier_cols = config.get("remove_outliers", [])
        total_outliers = 0
        for col in outlier_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Skip if IQR is zero (constant column) -- would remove all data
            if IQR == 0:
                report["notes"] += f"Skipped outlier removal for {col} (constant values). "
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            total_outliers += before - len(df)
        report["outliers_removed"] = total_outliers

        # 5. Drop columns that are entirely null
        cols_before = set(df.columns)
        df = df.dropna(axis=1, how="all")
        dropped_cols = list(cols_before - set(df.columns))
        report["columns_dropped"] = dropped_cols

        report["rows_after"] = len(df)

        # Guard: if cleaning removed all rows
        if df.empty:
            report["notes"] += "WARNING: All rows were removed during cleaning. "

        # Save cleaned file
        from pathlib import Path
        cleaned_path = str(Path(file_path).parent / f"cleaned_{Path(file_path).name}")
        df.to_csv(cleaned_path, index=False)

        result = {
            "cleaned_file_path": cleaned_path,
            "cleaning_report": report,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
