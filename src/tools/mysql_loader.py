"""
MySQLLoaderTool -- Designs relational schemas and loads data into MySQL.
Creates a dedicated database per CSV with agent-designed normalized tables.
"""

import json
import re
import pandas as pd
from crewai.tools import tool

from src.utils.security import sanitize_db_name, sanitize_table_name
from src.utils.db import create_database, get_db_engine


@tool("MySQL Schema Designer")
def mysql_schema_designer_tool(file_path: str, schema_design: str) -> str:
    """
    Analyze cleaned CSV data and create a normalized MySQL database with the agent-designed schema.
    The agent provides a relational schema design, and this tool creates the database, tables,
    and loads the data accordingly.

    Args:
        file_path: Absolute path to the cleaned CSV file.
        schema_design: JSON string describing the relational schema.
            Example: {
                "tables": [
                    {
                        "name": "customers",
                        "columns": {"id": "INT AUTO_INCREMENT", "name": "VARCHAR(255)", "email": "VARCHAR(255)"},
                        "primary_key": "id",
                        "source_columns": ["customer_name", "customer_email"]
                    },
                    {
                        "name": "orders",
                        "columns": {"id": "INT AUTO_INCREMENT", "customer_id": "INT", "amount": "DECIMAL(10,2)"},
                        "primary_key": "id",
                        "foreign_keys": {"customer_id": "customers(id)"},
                        "source_columns": ["customer_name", "order_amount"]
                    }
                ],
                "rationale": "Normalized customer data from repeated entries"
            }
            If no schema_design is provided or is "auto", a single table will be created from the CSV as-is.
    """
    engine = None
    try:
        df = pd.read_csv(file_path)
        original_filename = file_path.split("\\")[-1].split("/")[-1]

        # Guard: empty DataFrame
        if df.empty:
            return json.dumps({"error": "Cannot load empty DataFrame into MySQL."})

        # Sanitize column names for MySQL compatibility (no spaces, special chars)
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col).strip('_') for col in df.columns]
        # Handle duplicate column names after sanitization
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols

        # Create a dedicated database
        db_name = sanitize_db_name(original_filename)
        create_database(db_name)
        engine = get_db_engine(db_name)

        # Parse schema design
        try:
            design = json.loads(schema_design)
        except (json.JSONDecodeError, TypeError):
            design = None

        result_tables = []

        if design and "tables" in design and len(design["tables"]) > 0:
            # Agent-designed relational schema
            for table_spec in design["tables"]:
                table_name = sanitize_table_name(table_spec.get("name", "data"))
                source_cols = table_spec.get("source_columns", [])

                # Filter relevant columns from the DataFrame
                if source_cols:
                    available_cols = [c for c in source_cols if c in df.columns]
                    table_df = df[available_cols].drop_duplicates() if available_cols else df
                else:
                    table_df = df

                # Write to MySQL using pandas to_sql (handles type mapping)
                table_df.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists="replace",
                    index=False,
                    method="multi",
                    chunksize=1000,
                )

                result_tables.append({
                    "table_name": table_name,
                    "columns": list(table_df.columns),
                    "row_count": len(table_df),
                })

        else:
            # Auto mode: single table from CSV
            table_name = sanitize_table_name(original_filename.replace(".csv", ""))
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=1000,
            )
            result_tables.append({
                "table_name": table_name,
                "columns": list(df.columns),
                "row_count": len(df),
            })

        result = {
            "db_name": db_name,
            "tables": result_tables,
            "total_rows_loaded": sum(t["row_count"] for t in result_tables),
            "rationale": design.get("rationale", "Auto-loaded as single table") if design else "Auto-loaded as single table",
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if engine is not None:
            engine.dispose()
