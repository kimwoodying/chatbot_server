from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional

from django.utils import timezone

from chatbot.models import ChatMessage
from chatbot.services.tooling import (
    ToolContext,
    execute_tool,
    _extract_department,
    _extract_doctor_name,
    _is_holiday_date,
    CLINIC_CLOSED_REPLY,
    _format_doctor_reply_name,
)
from chatbot.services.intents.keywords import (
    TIME_EXTRACT_PATTERN,
    TIME_HINT_WORDS,
    ASAP_HINT_WORDS,
    DATE_EXTRACT_PATTERN,
    DAY_ONLY_PATTERN,
    DATE_KOR_PATTERN,
    DATE_SLASH_PATTERN,
    DATE_DASH_PATTERN,
    DATE_HINT_PATTERN,
    DOCTOR_SELECT_CUES,
    RESCHEDULE_TIME_KEEP_CUES,
    TIME_SPECIFIC_PATTERN,
    BOOKING_PROMPT_MARKERS,
)
from chatbot.services.intents.classifiers import (
    has_booking_intent,
    has_additional_booking_intent,
    has_reschedule_cue,
    match_symptom_department,
    has_time_hint,
)
from chatbot.services.flows import match_symptom_guide


def contains_asap(text: str) -> bool:
    if not text:
        return False
    return any(word in text for word in ASAP_HINT_WORDS)


def extract_date_phrase(text: str) -> str | None:
    if not text:
        return None
    match = DATE_EXTRACT_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return None


def extract_day_only(text: str) -> int | None:
    if not text:
        return None
    if "월" in text or "/" in text or "-" in text:
        return None
    match = DAY_ONLY_PATTERN.search(text)
    if not match:
        return None
    try:
        day = int(match.group(1))
    except (TypeError, ValueError):
        return None
    if 1 <= day <= 31:
        return day
    return None


def extract_day_only_list(text: str) -> list[int]:
    if not text:
        return []
    if "월" in text or "/" in text or "-" in text:
        return []
    matches = DAY_ONLY_PATTERN.findall(text)
    days: list[int] = []
    for value in matches:
        try:
            day = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= day <= 31 and day not in days:
            days.append(day)
    return days


def extract_numeric_day(text: str) -> int | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped.isdigit():
        return None
    try:
        day = int(stripped)
    except (TypeError, ValueError):
        return None
    if 1 <= day <= 31:
        return day
    return None


def is_multi_date_prompt(text: str) -> bool:
    if not text:
        return False
    return "여러 날짜" in text or "날짜를 하나만" in text


def build_date_from_base_day(base: date, day: int) -> date | None:
    year = base.year
    month = base.month
    if day < base.day:
        month += 1
        if month > 12:
            month = 1
            year += 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def build_date_same_month(base: date, day: int) -> date | None:
    try:
        return date(base.year, base.month, day)
    except ValueError:
        return None


def merge_date_with_time(preferred_time: str | None, date_hint: str | None) -> str | None:
    if not preferred_time or not date_hint:
        return preferred_time
    if has_specific_time(preferred_time) and not extract_date_phrase(preferred_time):
        return f"{date_hint} {preferred_time}"
    return preferred_time


def parse_date_only(text: str) -> date | None:
    if not text:
        return None
    match = DATE_KOR_PATTERN.search(text) or DATE_SLASH_PATTERN.search(text) or DATE_DASH_PATTERN.search(text)
    if not match:
        return None
    year_text = match.group(1)
    month_text = match.group(2)
    day_text = match.group(3)
    try:
        month = int(month_text)
        day = int(day_text)
    except (TypeError, ValueError):
        return None
    base = timezone.localdate()
    year = int(year_text) if year_text else base.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None
    if not year_text and candidate < base:
        candidate = date(year + 1, month, day)
    return candidate


def is_closed_clinic_date(value: date) -> bool:
    if _is_holiday_date(value):
        return True
    weekday = value.weekday()
    if weekday == 6:
        return True
    if weekday == 5:
        week_of_month = (value.day - 1) // 7 + 1
        return week_of_month not in {1, 3}
    return False


def maybe_reject_closed_date(text: str) -> str | None:
    date_phrase = extract_date_phrase(text)
    if not date_phrase:
        return None
    value = parse_date_only(date_phrase)
    if not value:
        return None
    if is_closed_clinic_date(value):
        return CLINIC_CLOSED_REPLY
    return None


def extract_time_phrase(text: str) -> str | None:
    if not text:
        return None
    parts = []
    seen = set()
    for match in TIME_EXTRACT_PATTERN.finditer(text):
        token = match.group(0).strip()
        if token and token not in seen:
            seen.add(token)
            parts.append(token)
    if parts:
        if not DATE_HINT_PATTERN.search(text):
            for word in TIME_HINT_WORDS:
                if word in text:
                    parts.insert(0, word)
                    break
        return " ".join(parts)
    for word in TIME_HINT_WORDS:
        if word in text:
            return word
    if contains_asap(text):
        return "가능한 빠른 시간"
    return None


def extract_numeric_hour(text: str) -> int | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped.isdigit():
        return None
    try:
        hour = int(stripped)
    except ValueError:
        return None
    if 0 <= hour <= 23:
        return hour
    return None


def has_specific_time(text: str) -> bool:
    if not text:
        return False
    return TIME_SPECIFIC_PATTERN.search(text) is not None


def has_date_hint(text: str) -> bool:
    if not text:
        return False
    if DATE_HINT_PATTERN.search(text):
        return True
    return any(word in text for word in TIME_HINT_WORDS)


def normalize_preferred_time(value: str | None, asap: bool) -> str | None:
    if not value:
        return "가능한 빠른 시간" if asap else None
    if has_specific_time(value):
        return value
    if asap:
        if "빠른" in value or "가능한" in value:
            return "가능한 빠른 시간"
        return f"{value} 가능한 빠른 시간"
    return value


def extract_selected_doctor_name(query: str, metadata: Dict[str, Any] | None) -> str | None:
    name = _extract_doctor_name(query, None)
    if name:
        return name
    if not isinstance(metadata, dict):
        return None
    if any(cue in query for cue in DOCTOR_SELECT_CUES):
        meta_name = metadata.get("doctor_name") or metadata.get("doctor") or metadata.get("doctorName")
        if isinstance(meta_name, str) and meta_name.strip():
            return meta_name.strip()
    return None


def infer_recent_doctor_name(session_id: str | None) -> str | None:
    if not session_id:
        return None
    recent_messages = list(
        ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:8]
    )
    for message in recent_messages:
        meta = message.metadata if isinstance(message.metadata, dict) else None
        if meta:
            for key in ("doctor_name", "doctor", "doctorName"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        answer = message.bot_answer or ""
        match = re.search(r"예약 의료진은\s*([가-힣0-9]{1,10})", answer)
        if match:
            return match.group(1)
        match = re.search(r"예약 의료진을\s*([가-힣0-9]{1,10})\s*(?:으로|로)\s*변경", answer)
        if match:
            return match.group(1)
    return None


def infer_recent_department(session_id: str | None) -> str | None:
    if not session_id:
        return None
    recent_messages = list(
        ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:6]
    )
    for message in recent_messages:
        for text in [message.user_question, message.bot_answer]:
            if not text:
                continue
            inferred = _extract_department(text, None)
            if inferred:
                return inferred
    return None


def infer_wait_department(tool_context: ToolContext | None) -> str | None:
    if not tool_context:
        return None
    result = execute_tool(
        "reservation_history",
        {"offset": 0, "limit": 1, "reply_style": "single", "label": "예약"},
        tool_context,
    )
    if not isinstance(result, dict):
        return None
    reservations = result.get("reservations") or []
    if not reservations:
        return None
    department = reservations[0].get("department")
    if isinstance(department, str) and department.strip():
        return department.strip()
    return None


def should_reschedule_from_summary(query: str, metadata: Dict[str, Any] | None) -> bool:
    if not query:
        return False
    if not has_booking_intent(query):
        return False
    if has_additional_booking_intent(query):
        return False
    if has_reschedule_cue(query):
        return True
    if match_symptom_department(query) or match_symptom_guide(query):
        return False
    has_time = has_time_hint(query) or any(
        marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
    )
    if has_time:
        return True
    return _extract_department(query, metadata) is not None


def build_time_followup_message(time_hint: str | None) -> str:
    if time_hint:
        return f"{time_hint} 기준으로 희망 시간대를 알려주세요."
    return "예약을 위해 희망 날짜/시간을 알려주세요."
