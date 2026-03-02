"""
Data Masking Utility -- Detects and masks PII (Personally Identifiable Information).
Prevents sensitive data from leaking into agent context, reports, or MySQL storage.
"""

import re
import hashlib
import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PII DETECTION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?"           # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-]?)?"        # optional area code
    r"\d{3,5}[\s\-]?\d{3,5}"           # main number
)

_SSN_RE = re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b")

_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")

_IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# Column-name heuristics (lowercase matching)
_PII_COLUMN_PATTERNS = {
    "email":   ["email", "e_mail", "e-mail", "email_address", "emailaddress"],
    "phone":   ["phone", "telephone", "mobile", "cell", "contact_number", "phone_number"],
    "name":    ["name", "first_name", "last_name", "full_name", "firstname", "lastname",
                "fullname", "patient_name", "customer_name", "employee_name", "user_name"],
    "ssn":     ["ssn", "social_security", "social_security_number", "sin"],
    "address": ["address", "street", "city", "zip", "zipcode", "zip_code", "postal",
                "postal_code", "state", "country"],
    "dob":     ["dob", "date_of_birth", "birth_date", "birthdate", "birthday"],
    "credit_card": ["credit_card", "card_number", "cc_number", "ccnumber"],
    "ip":      ["ip", "ip_address", "ipaddress"],
    "password": ["password", "passwd", "pwd", "secret", "token", "api_key"],
}


# ─────────────────────────────────────────────────────────────────────────────
# PII DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_by_column_name(col_name: str) -> Optional[str]:
    """Check if a column name matches known PII patterns."""
    clean = col_name.lower().strip()
    for pii_type, patterns in _PII_COLUMN_PATTERNS.items():
        if clean in patterns:
            return pii_type
    return None


def _detect_by_content(series: pd.Series, sample_size: int = 200) -> Optional[str]:
    """
    Sample column values and check for PII content patterns.
    Returns the detected PII type or None.
    """
    clean = series.dropna().astype(str)
    if clean.empty:
        return None

    sample = clean.head(sample_size)
    total = len(sample)
    if total == 0:
        return None

    # Count matches for each pattern
    email_hits = sum(1 for v in sample if _EMAIL_RE.search(v))
    phone_hits = sum(1 for v in sample if _PHONE_RE.fullmatch(v.strip()))
    ssn_hits = sum(1 for v in sample if _SSN_RE.fullmatch(v.strip()))
    cc_hits = sum(1 for v in sample if _CREDIT_CARD_RE.fullmatch(v.strip()))
    ip_hits = sum(1 for v in sample if _IP_RE.fullmatch(v.strip()))

    # Require >30% match rate to flag as PII
    threshold = 0.3

    if email_hits / total > threshold:
        return "email"
    if ssn_hits / total > threshold:
        return "ssn"
    if cc_hits / total > threshold:
        return "credit_card"
    if ip_hits / total > threshold:
        return "ip"
    if phone_hits / total > threshold:
        return "phone"

    return None


def detect_pii(df: pd.DataFrame) -> dict[str, str]:
    """
    Detect PII columns in a DataFrame.

    Returns:
        Dict mapping column name -> PII type (e.g., {"email_col": "email", "name": "name"})
    """
    pii_map = {}

    for col in df.columns:
        # First try column name heuristics (faster, more reliable)
        pii_type = _detect_by_column_name(col)
        if pii_type:
            pii_map[col] = pii_type
            continue

        # Then try content-based detection (for non-numeric columns only)
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            pii_type = _detect_by_content(df[col])
            if pii_type:
                pii_map[col] = pii_type

    return pii_map


# ─────────────────────────────────────────────────────────────────────────────
# MASKING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _hash_value(value: str, length: int = 8) -> str:
    """Generate a short deterministic hash for a value."""
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def _mask_email(value: str) -> str:
    """Mask an email: 'john.doe@example.com' -> 'j***@e***.com'"""
    if pd.isna(value):
        return value
    value = str(value)
    match = _EMAIL_RE.search(value)
    if not match:
        return f"[EMAIL_{_hash_value(value)[:6]}]"
    email = match.group()
    parts = email.split("@")
    if len(parts) == 2:
        local = parts[0][0] + "***" if parts[0] else "***"
        domain_parts = parts[1].split(".")
        domain = domain_parts[0][0] + "***" if domain_parts[0] else "***"
        ext = domain_parts[-1] if len(domain_parts) > 1 else "com"
        return f"{local}@{domain}.{ext}"
    return f"[EMAIL_{_hash_value(value)[:6]}]"


def _mask_phone(value: str) -> str:
    """Mask a phone: show only last 4 digits."""
    if pd.isna(value):
        return value
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 4:
        return f"***-***-{digits[-4:]}"
    return "[PHONE_REDACTED]"


def _mask_ssn(value: str) -> str:
    """Mask SSN: '123-45-6789' -> '***-**-6789'"""
    if pd.isna(value):
        return value
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 4:
        return f"***-**-{digits[-4:]}"
    return "[SSN_REDACTED]"


def _mask_credit_card(value: str) -> str:
    """Mask credit card: show only last 4 digits."""
    if pd.isna(value):
        return value
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 4:
        return f"****-****-****-{digits[-4:]}"
    return "[CC_REDACTED]"


def _mask_name(value: str) -> str:
    """Mask a name with a deterministic hash."""
    if pd.isna(value):
        return value
    return f"PERSON_{_hash_value(str(value))}"


def _mask_generic(value: str) -> str:
    """Generic masking for addresses, DOBs, IPs, passwords etc."""
    if pd.isna(value):
        return value
    return f"[REDACTED_{_hash_value(str(value))[:6]}]"


# Masking dispatch
_MASKERS = {
    "email": _mask_email,
    "phone": _mask_phone,
    "ssn": _mask_ssn,
    "credit_card": _mask_credit_card,
    "name": _mask_name,
    "address": _mask_generic,
    "dob": _mask_generic,
    "ip": _mask_generic,
    "password": _mask_generic,
}


def mask_dataframe(df: pd.DataFrame, pii_map: dict[str, str]) -> pd.DataFrame:
    """
    Apply masking to PII columns in a DataFrame.

    Args:
        df: The DataFrame to mask.
        pii_map: Dict from detect_pii() mapping column -> PII type.

    Returns:
        A new DataFrame with PII columns masked.
    """
    masked = df.copy()

    for col, pii_type in pii_map.items():
        if col not in masked.columns:
            continue
        masker = _MASKERS.get(pii_type, _mask_generic)
        masked[col] = masked[col].apply(lambda v: masker(v) if pd.notna(v) else v)

    return masked


def redact_sample_values(sample_values: list[str], pii_type: str) -> list[str]:
    """
    Redact sample values for agent-facing summaries.
    Instead of showing actual values, shows placeholder text.
    """
    placeholders = {
        "email": "[email detected]",
        "phone": "[phone number detected]",
        "ssn": "[SSN detected]",
        "credit_card": "[credit card detected]",
        "name": "[personal name detected]",
        "address": "[address detected]",
        "dob": "[date of birth detected]",
        "ip": "[IP address detected]",
        "password": "[sensitive credential detected]",
    }
    placeholder = placeholders.get(pii_type, "[PII detected]")
    return [placeholder] * min(len(sample_values), 3)


def get_masking_summary(pii_map: dict[str, str]) -> str:
    """Generate a human-readable summary of detected PII."""
    if not pii_map:
        return "No PII detected."

    lines = [f"Detected {len(pii_map)} PII column(s):"]
    for col, pii_type in pii_map.items():
        lines.append(f"  - {col}: {pii_type}")
    lines.append("These columns have been masked to prevent data leakage.")
    return "\n".join(lines)
