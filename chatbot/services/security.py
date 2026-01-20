from __future__ import annotations

import re
from typing import Any, Dict


ALLOWED_METADATA_KEYS = {
    "request_id",
    "user_id",
    "session_id",
    "account_id",
    "patient_id",
    "patient_identifier",
    "patient_pk",
    "patient_name",
    "patient_phone",
    "department",
    "preferred_time",
    "reservation_id",
    "channel",
    "tool_name",
    "tool_intent",
}

PHONE_PATTERN = re.compile(r"(?:0\d{1,2})[-\s]?\d{3,4}[-\s]?\d{4}")


def mask_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 7:
        return value
    return f"{digits[:3]}****{digits[-4:]}"


def mask_pii_text(text: str) -> str:
    def _mask(match: re.Match) -> str:
        return mask_phone(match.group(0))

    return PHONE_PATTERN.sub(_mask, text)


def sanitize_metadata_for_prompt(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in ALLOWED_METADATA_KEYS:
            continue
        if isinstance(value, str):
            if key in {"patient_phone"}:
                sanitized[key] = mask_phone(value)
            else:
                sanitized[key] = value.strip()
        else:
            sanitized[key] = value
    return sanitized


def mask_metadata_for_logs(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    masked: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            if "phone" in key or "tel" in key:
                masked[key] = mask_phone(value)
            else:
                masked[key] = value[:120]
        else:
            masked[key] = value
    return masked
