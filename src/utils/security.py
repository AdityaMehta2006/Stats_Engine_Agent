"""
Security utilities for file validation and sanitization.
Prevents malicious uploads, path traversal, and injection attacks.
"""

import os
import re
import uuid
import mimetypes
from pathlib import Path
from typing import Optional


# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_MIME_TYPES = {"text/csv", "text/plain", "application/vnd.ms-excel"}

# Max filename length after sanitization
MAX_FILENAME_LENGTH = 100


def validate_file_extension(filename: str) -> bool:
    """Check if file has an allowed extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def validate_mime_type(filename: str) -> bool:
    """Check MIME type based on file extension."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        return False
    return mime_type in ALLOWED_MIME_TYPES


def validate_file_size(file_size_bytes: int, max_size_mb: int) -> bool:
    """Check if file is within size limit."""
    max_bytes = max_size_mb * 1024 * 1024
    return file_size_bytes <= max_bytes


def sanitize_filename(filename: str) -> str:
    """
    Generate a safe filename using UUID prefix.
    Strips path components, special chars, and limits length.
    """
    # Extract just the filename (no directory traversal)
    base = os.path.basename(filename)

    # Remove everything except alphanumeric, hyphens, underscores, dots
    safe_name = re.sub(r"[^\w\-.]", "_", base)

    # Truncate if too long
    name_part = Path(safe_name).stem[:MAX_FILENAME_LENGTH]
    ext = Path(safe_name).suffix.lower()

    # Prefix with UUID for uniqueness
    unique_name = f"{uuid.uuid4().hex[:8]}_{name_part}{ext}"
    return unique_name


def sanitize_db_name(raw_name: str) -> str:
    """
    Create a safe MySQL database name from a CSV filename.
    Format: csv_{sanitized_name}_{short_uuid}
    """
    # Strip extension and path
    base = Path(raw_name).stem

    # Keep only alphanumeric and underscores, lowercase
    safe = re.sub(r"[^a-z0-9_]", "_", base.lower())

    # Remove consecutive underscores
    safe = re.sub(r"_+", "_", safe).strip("_")

    # Truncate and add prefix + uniqueness
    safe = safe[:40]
    short_id = uuid.uuid4().hex[:6]
    return f"csv_{safe}_{short_id}"


def sanitize_table_name(raw_name: str) -> str:
    """Create a safe MySQL table name."""
    safe = re.sub(r"[^a-z0-9_]", "_", raw_name.lower())
    safe = re.sub(r"_+", "_", safe).strip("_")

    # Table names can't start with a number
    if safe and safe[0].isdigit():
        safe = f"t_{safe}"

    return safe[:64]  # MySQL table name limit


def validate_upload(
    filename: str,
    file_size_bytes: int,
    max_size_mb: int,
) -> Optional[str]:
    """
    Full validation of an uploaded file.
    Returns None if valid, or an error message string if invalid.
    """
    if not filename:
        return "No filename provided."

    if not validate_file_extension(filename):
        return f"Invalid file type. Only CSV files are allowed."

    if not validate_mime_type(filename):
        return f"Invalid MIME type for file '{filename}'."

    if not validate_file_size(file_size_bytes, max_size_mb):
        return f"File exceeds maximum size of {max_size_mb} MB."

    return None
