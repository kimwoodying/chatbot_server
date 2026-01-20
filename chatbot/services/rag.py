# chatbot/services/rag.py
from __future__ import annotations

import json
import logging
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from chatbot.config import get_settings
from chatbot.services.embeddings import embed_texts
from chatbot.services.vector_store import get_vector_store
from chatbot.services.cache_service import (
    CACHE_SCOPE_QUERY_ONLY,
    get_cached_response,
    hash_text,
    save_cache_response,
)
from chatbot.services.gemini_client import call_gemini_with_rag
from chatbot.services.safety import build_safety_response
from chatbot.services.static_answers import (
    collect_contact_numbers,
    extract_context_text,
    get_static_answer,
)
from chatbot.models import ChatMessage

from chatbot.services.common import (
    DEFAULT_GREETING_REPLY,
    AUTH_REQUIRED_REPLY,
    AUTH_METADATA_KEYS,
    clean_response,
    ToolReply,
)
from chatbot.services.intents.classifiers import (
    is_smalltalk_query,
    is_fixed_info_query,
    is_doctor_followup,
    is_doctor_department_followup,
    match_symptom_department,
    has_time_hint,
    has_booking_intent,
    has_symptom_time_booking_intent,
    has_reschedule_cue,
    has_doctor_change_cue,
    has_cancel_cue,
    has_bulk_cancel_cue,
    needs_reservation_login_guard,
    is_wait_department_prompt,
    is_symptom_department_request,
)
from chatbot.services.intents.keywords import *
from chatbot.services.flows import (
    handle_medical_history,
    match_symptom_guide,
)
from chatbot.services.reservation_flow import (
    handle_reservation_followup,
)
from chatbot.services.extraction import (
    infer_recent_doctor_name,
    infer_recent_department,
    infer_wait_department,
    extract_time_phrase,
    normalize_preferred_time,
    maybe_reject_closed_date,
    has_specific_time,
    contains_asap,
    build_time_followup_message,
)
from chatbot.services.tooling import (
    ToolContext,
    build_tool_context,
    build_slot_fill_response,
    classify_tool_intent,
    execute_tool,
    should_use_tools,
    TIME_REQUIRED_REPLY,
    CLINIC_CLOSED_REPLY,
    _build_doctor_table,
    _format_doctor_display_name,
    _format_doctor_reply_name,
    _extract_department,
    _extract_preferred_time,
    _extract_doctor_name,
    _is_holiday_date,
    _is_doctor_query,
    _has_auth_context,
)
from django.utils import timezone

logger = logging.getLogger(__name__)




def _build_sources_hash(metas: List[Dict[str, Any]]) -> str:
    if not metas:
        return ""
    parts = []
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        raw_id = meta.get("id") or meta.get("chunk_id") or meta.get("doc_id") or ""
        source = meta.get("source") or meta.get("title") or ""
        parts.append(f"{raw_id}:{source}")
    joined = "|".join(sorted(p for p in parts if p))
    return hash_text(joined)


def _build_sources(metas: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for meta, score in zip(metas, scores):
        if not isinstance(meta, dict):
            continue
        source_id = meta.get("id") or meta.get("chunk_id") or meta.get("doc_id")
        title = meta.get("title") or meta.get("source_file") or meta.get("source") or ""
        snippet = meta.get("snippet") or meta.get("text") or meta.get("chunk") or ""
        if isinstance(snippet, str) and len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "..."
        sources.append(
            {
                "type": "rag",
                "id": source_id,
                "title": title,
                "source": meta.get("source") or meta.get("source_file") or "",
                "category": meta.get("category") or "",
                "score": score,
                "snippet": snippet,
            }
        )
    return sources



def _keyword_filter(user_message: str, results: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
    """
    아주 단순한 키워드 기반 필터:
    - '금식' 질문이면 '금식'이 들어간 문서 chunk를 우선 사용
    - '주차' 질문이면 '주차'가 들어간 문서 chunk를 우선 사용
    """
    if not user_message or not results:
        return results

    # 우선순위를 줄 키워드들
    boost_keywords = ["금식", "주차", "예약", "위치", "면회", "서류", "비용"]
    
    # 질문에 포함된 키워드 추출
    active_keywords = [k for k in boost_keywords if k in user_message]
    if not active_keywords:
        return results

    boosted = []
    others = []

    for dist, meta in results:
        text = extract_context_text(meta).lower()
        if any(k in text for k in active_keywords):
            # 거리를 약간 줄여서 우선순위 높임 (유사도 증가 효과)
            boosted.append((dist * 0.8, meta))
        else:
            others.append((dist, meta))

    return sorted(boosted + others, key=lambda x: x[0])


def _get_parking_keywords(user_message: str) -> list[str]:
    q = user_message or ""
    q_lower = q.lower()
    if any(k in q for k in ["주차", "정산", "정산소"]) or "parking" in q_lower:
        return [
            "주차",
            "주차장",
            "주차요금",
            "주차 요금",
            "주차비",
            "면회객",
            "상주",
            "정산",
            "정산소",
            "무료주차",
            "무료 주차",
        ]
    return []


TRUSTED_INFO_SOURCE_FILES = ("hospital_info.txt", "parking_info.txt")





def _filter_trusted_info_meta(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trusted: List[Dict[str, Any]] = []
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        source_file = str(meta.get("source_file") or "")
        source = str(meta.get("source") or "")
        title = str(meta.get("title") or "")
        haystack = " ".join([source_file, source, title])
        if any(name in haystack for name in TRUSTED_INFO_SOURCE_FILES):
            trusted.append(meta)
    return trusted
def _strip_parking_settlement_info(contexts_text: List[str], query: str) -> List[str]:
    if not contexts_text:
        return contexts_text
    q = query or ""
    if any(k in q for k in ["정산", "정산소"]):
        return contexts_text

    cleaned: list[str] = []
    for ctx in contexts_text:
        lines = [line for line in ctx.splitlines() if "정산" not in line]
        filtered = "\n".join(line for line in lines if line.strip())
        if filtered.strip():
            cleaned.append(filtered)
    return cleaned or contexts_text














def _has_reschedule_cue(query: str) -> bool:
    return any(k in query for k in RESCHEDULE_CUES)

def _has_doctor_change_cue(query: str) -> bool:
    if not query:
        return False
    return any(k in query for k in DOCTOR_CHANGE_CUES)


def _is_doctor_change_prompt(text: str) -> bool:
    if not text:
        return False
    return DOCTOR_CHANGE_PROMPT in text


def _is_doctor_select_prompt(text: str) -> bool:
    if not text:
        return False
    if _is_doctor_change_prompt(text):
        return False
    return DOCTOR_SELECT_PROMPT in text


def _extract_selected_doctor_name(query: str, metadata: Dict[str, Any] | None) -> str | None:
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


def _infer_recent_doctor_name(session_id: str | None) -> str | None:
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
        match = re.search(r"예약 의료진은\\s*([가-힣0-9]{1,10})", answer)
        if match:
            return match.group(1)
        match = re.search(r"예약 의료진을\\s*([가-힣0-9]{1,10})\\s*(?:으로|로)\\s*변경", answer)
        if match:
            return match.group(1)
    return None



    if not query:
        return False
    if not has_booking_intent(query):
        return False
    if _has_additional_booking_intent(query):
        return False
    if _has_reschedule_cue(query):
        return True
    if match_symptom_department(query) or match_symptom_guide(query):
        return False
    has_time_hint = _has_time_or_date_hint(query) or any(
        marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
    )
    if has_time_hint:
        return True
    return _extract_department(query, metadata) is not None


def _has_cancel_cue(query: str) -> bool:
    return any(k in query for k in CANCEL_CUES)


def _is_negative_reply(query: str) -> bool:
    if not query:
        return False
    compact = re.sub(r"\s+", "", query.lower())
    return any(cue in compact for cue in NEGATIVE_CUES)


def _is_negative_only_reply(query: str, metadata: Dict[str, Any] | None) -> bool:
    if not _is_negative_reply(query):
        return False
    if _has_time_or_date_hint(query):
        return False
    if _extract_department(query, metadata):
        return False
    if match_symptom_department(query) or match_symptom_guide(query):
        return False
    if _extract_doctor_name(query, metadata):
        return False
    if _has_reschedule_cue(query) or _has_cancel_cue(query) or _has_doctor_change_cue(query):
        return False
    return True


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


def _infer_recent_department(session_id: str | None) -> str | None:
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


def _has_bulk_cancel_cue(query: str) -> bool:
    if not query or not _has_cancel_cue(query):
        return False
    if any(word in query for word in BULK_CANCEL_CUES):
        return True
    compact = query.replace(" ", "")
    if any(phrase in compact for phrase in ["다취소", "전부취소", "모두취소", "전체취소", "일괄취소"]):
        return True
    if " 다" in query:
        return True
    return False


def _has_department_confirmation_cue(query: str) -> bool:
    if not query:
        return False
    department = _extract_department(query, None)
    if not department:
        return False
    text = query.strip()
    return any(token in text for token in ["야", "맞아", "맞나요", "맞지", "인가"])


def _is_reservation_summary(text: str) -> bool:
    if not text or "예약" not in text:
        return False
    if any(marker in text for marker in BOOKING_PROMPT_MARKERS):
        return False
    if any(
        cue in text
        for cue in [
            "예약 가능 시간",
            "운영 시간",
            "공휴일",
            "평일",
            "토요일",
            "시간으로 알려주세요",
            "희망 날짜/시간",
            "희망 시간",
            "희망 날짜",
        ]
    ):
        return False
    if TIME_EXTRACT_PATTERN.search(text):
        return True
    return any(marker in text for marker in ["예약은", "예약이", "예약입니다", "진료예요", "진료입니다"])


def _has_time_or_date_hint(text: str) -> bool:
    if not text:
        return False
    if TIME_EXTRACT_PATTERN.search(text):
        return True
    if DAY_ONLY_PATTERN.search(text):
        return True
    return any(word in text for word in TIME_HINT_WORDS)

def contains_asap(text: str) -> bool:
    if not text:
        return False
    return any(word in text for word in ASAP_HINT_WORDS)

def _extract_date_phrase(text: str) -> str | None:
    if not text:
        return None
    match = DATE_EXTRACT_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return None


def _extract_day_only(text: str) -> int | None:
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







def _extract_numeric_day(text: str) -> int | None:
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


def _is_multi_date_prompt(text: str) -> bool:
    if not text:
        return False
    return "여러 날짜" in text or "날짜를 하나만" in text

def _build_date_from_base_day(base: date, day: int) -> date | None:
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


def _build_date_same_month(base: date, day: int) -> date | None:
    try:
        return date(base.year, base.month, day)
    except ValueError:
        return None


def _merge_date_with_time(preferred_time: str | None, date_hint: str | None) -> str | None:
    if not preferred_time or not date_hint:
        return preferred_time
    if has_specific_time(preferred_time) and not _extract_date_phrase(preferred_time):
        return f"{date_hint} {preferred_time}"
    return preferred_time


def _parse_date_only(text: str) -> date | None:
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


def _is_closed_clinic_date(value: date) -> bool:
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
    date_phrase = _extract_date_phrase(text)
    if not date_phrase:
        return None
    value = _parse_date_only(date_phrase)
    if not value:
        return None
    if _is_closed_clinic_date(value):
        return CLINIC_CLOSED_REPLY
    return None



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


def _extract_numeric_hour(text: str) -> int | None:
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


def _has_date_hint(text: str) -> bool:
    if not text:
        return False
    if DATE_HINT_PATTERN.search(text):
        return True
    return any(word in text for word in TIME_HINT_WORDS)


def _is_booking_prompt(text: str) -> bool:
    if not text:
        return False
    return any(marker in text for marker in BOOKING_PROMPT_MARKERS)


def _has_auth_context(metadata: Dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return any(metadata.get(key) for key in AUTH_METADATA_KEYS)


def needs_reservation_login_guard(query: str) -> bool:
    if not query:
        return False
    return any(cue in query for cue in RESERVATION_LOGIN_GUARD_CUES)


def build_tool_context(session_id: str | None, metadata: Dict[str, Any] | None) -> ToolContext:
    request_id = metadata.get("request_id") if isinstance(metadata, dict) else None
    user_id = None
    if isinstance(metadata, dict):
        user_id = (
            metadata.get("user_id")
            or metadata.get("patient_id")
            or metadata.get("patientId")
            or metadata.get("account_id")
            or metadata.get("auth_user_id")
        )
    return ToolContext(
        session_id=session_id,
        metadata=metadata,
        request_id=request_id,
        user_id=user_id,
    )


def build_time_followup_message(time_hint: str | None) -> str:
    if time_hint:
        return f"{time_hint} 기준으로 희망 시간대를 알려주세요."
    return "예약을 위해 희망 날짜/시간을 알려주세요."


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


def _is_wait_department_prompt(text: str) -> bool:
    if not text:
        return False
    markers = [
        "대기 현황을 확인할 진료과",
        "진료과를 알려주시면 대기 현황",
        "대기 현황을 확인해 드리겠습니다",
    ]
    return any(marker in text for marker in markers)



    if not session_id:
        return None
    department: str | None = None
    preferred_time: str | None = None
    date_hint: str | None = None
    asap = False
    if not _has_auth_context(metadata):
        if (
            any(cue in query for cue in RESERVATION_HISTORY_CUES)
            or _has_cancel_cue(query)
            or _has_reschedule_cue(query)
            or _has_doctor_change_cue(query)
        ):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
    if has_booking_intent(query):
        closed_reply = maybe_reject_closed_date(query)
        if closed_reply:
            return {"reply": closed_reply, "sources": []}
    if any(cue in query for cue in RESERVATION_HISTORY_CUES):
        if not _has_auth_context(metadata):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        has_time_hint = _has_time_or_date_hint(query) or any(
            marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
        )
        has_explicit_department = _extract_department(query, None) is not None
        if not (
            has_time_hint
            or _has_reschedule_cue(query)
            or _has_doctor_change_cue(query)
            or has_explicit_department
        ):
            tool_context = build_tool_context(session_id, metadata)
            result = execute_tool("reservation_history", {}, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                payload = {"reply": result["reply_text"], "sources": []}
                if result.get("table"):
                    payload["table"] = result["table"]
                return payload
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 예약 내역을 확인하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                    "sources": [],
                }
            return None
    create_from_summary = False
    last_message = (
        ChatMessage.objects.filter(session_id=session_id)
        .order_by("-created_at")
        .first()
    )
    if not last_message:
        return None
    last_bot_answer = last_message.bot_answer or ""
    if _is_wait_department_prompt(last_bot_answer):
        department = _extract_department(query, metadata)
        if department:
            tool_context = build_tool_context(session_id, metadata)
            result = execute_tool("wait_status", {"department": department}, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
        return {"reply": "대기 현황을 확인할 진료과를 알려주세요.", "sources": []}
    if (
        _is_negative_only_reply(query, metadata)
        and any(
            marker in last_bot_answer
            for marker in ["지난 날짜나 시간", "오늘 이후의 날짜와 시간"]
        )
    ):
        return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
    if _is_multi_date_prompt(last_bot_answer):
        if _is_negative_only_reply(query, metadata):
            return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
        day_only = _extract_day_only(query) or _extract_numeric_day(query)
        if not day_only:
            return {
                "reply": "여러 날짜가 있습니다. 예약할 날짜를 하나만 알려주세요.",
                "sources": [],
            }
        recent_messages = list(
            ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:6]
        )
        day_candidates: list[int] = []
        for text in [m.user_question for m in recent_messages]:
            day_candidates.extend(_extract_day_only_list(text))
        if day_only and day_only not in day_candidates:
            day_candidates.append(day_only)
        day_candidates = sorted(set(day_candidates))
        date_hint = None
        for text in [m.user_question for m in recent_messages] + [m.bot_answer for m in recent_messages]:
            date_hint = _extract_date_phrase(text)
            if date_hint:
                break
        base_date = _parse_date_only(date_hint) if date_hint else timezone.localdate()
        adjusted = (
            _build_date_same_month(base_date, day_only)
            or _build_date_from_base_day(base_date, day_only)
        ) if base_date else None
        if adjusted:
            date_hint = f"{adjusted.month}월 {adjusted.day}일"
        preferred_time = extract_time_phrase(query)
        if not preferred_time:
            numeric_hour = _extract_numeric_hour(query)
            if numeric_hour is not None:
                preferred_time = f"{numeric_hour}시"
        if not preferred_time:
            for text in [m.user_question for m in recent_messages]:
                preferred_time = extract_time_phrase(text)
                if not preferred_time:
                    numeric_hour = _extract_numeric_hour(text)
                    if numeric_hour is not None:
                        preferred_time = f"{numeric_hour}시"
                if preferred_time:
                    break
        preferred_time = _merge_date_with_time(preferred_time, date_hint)
        preferred_time = normalize_preferred_time(preferred_time, False)
        department = _extract_department(query, metadata) or _infer_recent_department(session_id)
        if not department:
            return {"reply": "예약을 위해 진료과를 알려주세요.", "sources": []}
        if not _has_auth_context(metadata):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        doctor_name = _extract_selected_doctor_name(query, metadata) or _extract_doctor_name(query, metadata)
        if not doctor_name:
            for message in recent_messages:
                for text in [message.user_question, message.bot_answer]:
                    doctor_name = _extract_doctor_name(text, metadata)
                    if doctor_name:
                        break
                if doctor_name:
                    break
        if not doctor_name:
            doctor_name = _infer_recent_doctor_name(session_id)
        if not doctor_name:
            tool_context = build_tool_context(session_id, metadata)
            doctor_result = execute_tool("doctor_list", {"department": department}, tool_context)
            payload = {
                "reply": f"{department} 의료진을 선택해 주세요. 선택 후 예약을 진행합니다.",
                "sources": [],
            }
            if isinstance(doctor_result, dict) and doctor_result.get("table"):
                payload["table"] = doctor_result["table"]
            return payload
        if not preferred_time:
            return {
                "reply": f"{doctor_name} 의료진으로 예약을 진행합니다. 희망 날짜/시간을 알려주세요.",
                "sources": [],
            }
        if not has_specific_time(preferred_time):
            return {
                "reply": f"{doctor_name} 의료진으로 예약을 진행합니다. {build_time_followup_message(preferred_time)}",
                "sources": [],
            }
        tool_context = build_tool_context(session_id, metadata)
        result = execute_tool(
            "reservation_create",
            {"department": department, "preferred_time": preferred_time, "doctor_name": doctor_name},
            tool_context,
        )
        if isinstance(result, dict) and result.get("reply_text"):
            reply_text = result["reply_text"]
            remaining_days = [d for d in day_candidates if d != day_only]
            if remaining_days:
                remain_text = ", ".join(f"{day}일" for day in remaining_days)
                reply_text = (
                    f"{reply_text} 남은 날짜({remain_text})도 예약할까요? 원하시면 날짜만 알려주세요."
                )
            payload = {"reply": reply_text, "sources": []}
            if result.get("table"):
                payload["table"] = result["table"]
            return payload
        if isinstance(result, dict) and result.get("status") == "ok":
            return {
                "reply": f"{department} 진료 예약 요청이 접수되었습니다. 희망 일정은 {preferred_time}입니다.",
                "sources": [],
            }
        if isinstance(result, dict) and result.get("status") == "error":
            return {
                "reply": "현재 예약을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                "sources": [],
            }
    if not _has_auth_context(metadata):
        if _is_doctor_change_prompt(last_bot_answer) or _is_reservation_summary(last_bot_answer):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
    if _is_doctor_change_prompt(last_bot_answer):
        if _is_negative_only_reply(query, metadata):
            return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
        tool_context = build_tool_context(session_id, metadata)
        doctor_name = _extract_selected_doctor_name(query, metadata)
        if not doctor_name:
            department = _extract_department(query, metadata) or _infer_recent_department(session_id)
            prompt = f"{department} {DOCTOR_CHANGE_PROMPT}" if department else DOCTOR_CHANGE_PROMPT
            return {"reply": prompt, "sources": []}
        result = execute_tool(
            "reservation_reschedule",
            {"doctor_name": doctor_name},
            tool_context,
        )
        if isinstance(result, dict) and result.get("reply_text"):
            payload = {"reply": result["reply_text"], "sources": []}
            if result.get("table"):
                payload["table"] = result["table"]
            return payload
        if isinstance(result, dict) and result.get("status") == "not_found":
            return {
                "reply": "변경할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.",
                "sources": [],
            }
        if isinstance(result, dict) and result.get("status") == "error":
            return {
                "reply": "현재 예약 변경을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                "sources": [],
            }
    if _is_doctor_select_prompt(last_bot_answer):
        if not _has_auth_context(metadata):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        if _has_cancel_cue(query):
            tool_context = build_tool_context(session_id, metadata)
            cancel_args = {"cancel_all": True} if _has_bulk_cancel_cue(query) else {}
            cancel_args["cancel_text"] = query
            result = execute_tool("reservation_cancel", cancel_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "취소할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 시스템에서 확인이 어렵습니다. 예약 번호나 연락처를 알려주시면 확인해 드리겠습니다.",
                    "sources": [],
                }
        if _is_negative_only_reply(query, metadata):
            return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
        tool_context = build_tool_context(session_id, metadata)
        doctor_name = _extract_selected_doctor_name(query, metadata)
        if not doctor_name:
            department = (
                _extract_department(query, metadata)
                or _extract_department(last_bot_answer, None)
                or _infer_recent_department(session_id)
            )
            doctor_result = None
            if department:
                doctor_result = execute_tool(
                    "doctor_list",
                    {"department": department},
                    tool_context,
                )
                if isinstance(doctor_result, dict) and doctor_result.get("status") in {"not_found", "error"}:
                    return {
                        "reply": doctor_result.get("reply_text")
                        or "해당 진료과 의료진 정보를 찾지 못했습니다. 원하시면 진료과명을 정확히 알려주세요.",
                        "sources": [],
                    }
            reply_text = f"{department} {DOCTOR_SELECT_PROMPT}" if department else DOCTOR_SELECT_PROMPT
            payload = {"reply": reply_text, "sources": []}
            if isinstance(doctor_result, dict) and doctor_result.get("table"):
                payload["table"] = doctor_result["table"]
            return payload
        recent_messages = list(
            ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:5]
        )
        department = (
            _extract_department(query, metadata)
            or _extract_department(last_bot_answer, None)
            or _infer_recent_department(session_id)
        )
        preferred_time = extract_time_phrase(query)
        if not preferred_time:
            numeric_hour = _extract_numeric_hour(query)
            if numeric_hour is not None:
                preferred_time = f"{numeric_hour}시"
        asap = contains_asap(query)
        if not department:
            search_texts = [last_message.bot_answer, last_message.user_question]
            search_texts.extend(m.user_question for m in recent_messages)
            for text in search_texts:
                if not text:
                    continue
                department = _extract_department(text, None) or match_symptom_department(text)
                if department:
                    break
        if not preferred_time:
            for text in [last_message.user_question] + [m.user_question for m in recent_messages]:
                if not text:
                    continue
                preferred_time = extract_time_phrase(text)
                if not preferred_time:
                    numeric_hour = _extract_numeric_hour(text)
                    if numeric_hour is not None:
                        preferred_time = f"{numeric_hour}시"
                asap = asap or contains_asap(text)
                if preferred_time:
                    break
        date_hint = _extract_date_phrase(last_bot_answer)
        if not date_hint:
            search_texts = [last_message.user_question] + [m.user_question for m in recent_messages]
            search_texts.extend(m.bot_answer for m in recent_messages)
            for text in search_texts:
                date_hint = _extract_date_phrase(text)
                if date_hint:
                    break
        day_only = _extract_day_only(query)
        if not day_only and _is_multi_date_prompt(last_bot_answer):
            day_only = _extract_numeric_day(query)
        day_candidates: list[int] = []
        for text in [last_message.user_question] + [m.user_question for m in recent_messages]:
            day_candidates.extend(_extract_day_only_list(text))
        day_candidates = sorted(set(day_candidates))
        if not date_hint and len(day_candidates) > 1 and not day_only:
            return {
                "reply": "여러 날짜가 있습니다. 예약할 날짜를 하나만 알려주세요.",
                "sources": [],
            }
        if not date_hint and not day_only and len(day_candidates) == 1:
            day_only = day_candidates[0]
        if day_only and not _extract_date_phrase(query):
            base_date = _parse_date_only(date_hint) if date_hint else timezone.localdate()
            adjusted = (
                _build_date_same_month(base_date, day_only)
                or _build_date_from_base_day(base_date, day_only)
            ) if base_date else None
            if adjusted:
                date_hint = f"{adjusted.month}월 {adjusted.day}일"
                if preferred_time and _extract_date_phrase(preferred_time):
                    preferred_time = date_hint
                else:
                    preferred_time = _merge_date_with_time(preferred_time, date_hint)
                    if not preferred_time:
                        preferred_time = date_hint
        preferred_time = _merge_date_with_time(preferred_time, date_hint)
        preferred_time = normalize_preferred_time(preferred_time, asap)
        closed_reply = maybe_reject_closed_date(preferred_time or "")
        if closed_reply:
            return {"reply": closed_reply, "sources": []}
        if not department:
            return {"reply": "예약을 위해 진료과를 알려주세요.", "sources": []}
        if not preferred_time:
            return {
                "reply": f"{doctor_name} 의료진으로 예약을 진행합니다. 희망 날짜/시간을 알려주세요.",
                "sources": [],
            }
        if not has_specific_time(preferred_time) and not asap:
            return {
                "reply": f"{doctor_name} 의료진으로 예약을 진행합니다. {build_time_followup_message(preferred_time)}",
                "sources": [],
            }
        result = execute_tool(
            "reservation_create",
            {"department": department, "preferred_time": preferred_time, "doctor_name": doctor_name},
            tool_context,
        )
        if isinstance(result, dict):
            if result.get("reply_text"):
                payload = {"reply": result["reply_text"], "sources": []}
                if result.get("table"):
                    payload["table"] = result["table"]
                return payload
            if result.get("status") == "ok":
                return {
                    "reply": f"{department} 진료 예약 요청이 접수되었습니다. 희망 일정은 {preferred_time}입니다.",
                    "sources": [],
                }
            if result.get("status") == "not_found":
                return {
                    "reply": "요청하신 의료진 정보를 찾지 못했습니다. 의료진 이름을 다시 알려주세요.",
                    "sources": [],
                }
            if result.get("status") == "error":
                return {
                    "reply": "현재 예약을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                    "sources": [],
                }
        return {
            "reply": "예약 처리 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "sources": [],
        }
    if "의료진으로 변경" in last_bot_answer:
        tool_context = build_tool_context(session_id, metadata)
        department = _extract_department(query, metadata)
        if not department:
            return {
                "reply": "어느 진료과 의료진으로 변경할까요?",
                "sources": [],
            }
        doctor_result = execute_tool(
            "doctor_list",
            {"department": department},
            tool_context,
        )
        payload = {"reply": f"{department} {DOCTOR_CHANGE_PROMPT}", "sources": []}
        if isinstance(doctor_result, dict) and doctor_result.get("table"):
            payload["table"] = doctor_result["table"]
        return payload
    if _has_doctor_change_cue(query):
        tool_context = build_tool_context(session_id, metadata)
        doctor_name = _extract_selected_doctor_name(query, metadata)
        if doctor_name:
            result = execute_tool(
                "reservation_reschedule",
                {"doctor_name": doctor_name},
                tool_context,
            )
            if isinstance(result, dict) and result.get("reply_text"):
                payload = {"reply": result["reply_text"], "sources": []}
                if result.get("table"):
                    payload["table"] = result["table"]
                return payload
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "변경할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 예약 변경을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                    "sources": [],
                }
        department = _extract_department(query, metadata) or _infer_recent_department(session_id)
        current_doctor = None
        current_department = None
        history = execute_tool(
            "reservation_history",
            {"offset": 0, "limit": 1, "reply_style": "single", "label": "예약"},
            tool_context,
        )
        if isinstance(history, dict) and history.get("reservations"):
            first = history["reservations"][0]
            current_department = first.get("department") or None
            raw_doctor = first.get("doctor_name") or None
            if raw_doctor:
                current_doctor = _format_doctor_reply_name(raw_doctor)
        department = department or current_department
        if not department:
            if current_doctor:
                return {
                    "reply": f"현재 예약 의료진은 {current_doctor}입니다. 어느 진료과 의료진으로 변경할까요?",
                    "sources": [],
                }
            return {
                "reply": "어느 진료과 의료진으로 변경할까요?",
                "sources": [],
            }
        doctor_result = execute_tool(
            "doctor_list",
            {"department": department},
            tool_context,
        )
        reply_text = f"{department} {DOCTOR_CHANGE_PROMPT}"
        if current_doctor:
            reply_text = f"현재 예약 의료진은 {current_doctor}입니다. {department} {DOCTOR_CHANGE_PROMPT}"
        payload = {"reply": reply_text, "sources": []}
        if isinstance(doctor_result, dict) and doctor_result.get("table"):
            payload["table"] = doctor_result["table"]
        return payload
    if _is_reservation_summary(last_bot_answer):
        tool_context = build_tool_context(session_id, metadata)
        auto_reschedule = _should_reschedule_from_summary(query, metadata)
        if _has_cancel_cue(query):
            cancel_args = {"cancel_all": True} if _has_bulk_cancel_cue(query) else {}
            cancel_args["cancel_text"] = query
            result = execute_tool("reservation_cancel", cancel_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {"reply": "취소할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.", "sources": []}
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 시스템에서 확인이 어렵습니다. 예약 번호나 연락처를 알려주시면 확인해 드리겠습니다.",
                    "sources": [],
                }
        if _has_reschedule_cue(query) or auto_reschedule:
            new_department = _extract_department(query, metadata)
            has_time_hint = _has_time_or_date_hint(query) or any(
                marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
            )
            if not has_time_hint and not new_department:
                return {
                    "reply": "예약 변경을 위해 변경할 날짜/시간이나 진료과를 알려주세요.",
                    "sources": [],
                }
            reschedule_args: Dict[str, Any] = {}
            if has_time_hint:
                new_time_text = query
                if _has_time_or_date_hint(query):
                    time_phrase = extract_time_phrase(query)
                    if time_phrase:
                        new_time_text = time_phrase
                recent_messages = list(
                    ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:5]
                )
                date_hint = _extract_date_phrase(last_bot_answer)
                if not date_hint:
                    search_texts = [last_message.user_question] + [m.user_question for m in recent_messages]
                    search_texts.extend(m.bot_answer for m in recent_messages)
                    for text in search_texts:
                        date_hint = _extract_date_phrase(text)
                        if date_hint:
                            break
                day_only = _extract_day_only(query)
                if not day_only and _is_multi_date_prompt(last_bot_answer):
                    day_only = _extract_numeric_day(query)
                day_candidates: list[int] = []
                for text in [last_message.user_question] + [m.user_question for m in recent_messages]:
                    day_candidates.extend(_extract_day_only_list(text))
                day_candidates = sorted(set(day_candidates))
                if not date_hint and len(day_candidates) > 1 and not day_only:
                    return {
                        "reply": "여러 날짜가 있습니다. 예약할 날짜를 하나만 알려주세요.",
                        "sources": [],
                    }
                if not date_hint and not day_only and len(day_candidates) == 1:
                    day_only = day_candidates[0]
                if day_only and not _extract_date_phrase(query):
                    base_date = _parse_date_only(date_hint) if date_hint else timezone.localdate()
                    adjusted = (
                        _build_date_same_month(base_date, day_only)
                        or _build_date_from_base_day(base_date, day_only)
                    ) if base_date else None
                    if adjusted:
                        date_hint = f"{adjusted.month}월 {adjusted.day}일"
                        if _extract_date_phrase(new_time_text or ""):
                            new_time_text = date_hint
                        else:
                            new_time_text = _merge_date_with_time(new_time_text, date_hint)
                            if not new_time_text:
                                new_time_text = date_hint
                new_time_text = _merge_date_with_time(new_time_text, date_hint)
                reschedule_args["new_time"] = normalize_preferred_time(new_time_text, False)
            if new_department:
                reschedule_args["new_department"] = new_department
            if not reschedule_args:
                return {
                    "reply": "예약 변경을 위해 변경할 날짜/시간이나 진료과를 알려주세요.",
                    "sources": [],
                }
            result = execute_tool("reservation_reschedule", reschedule_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "변경할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 예약 변경을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                    "sources": [],
                }

    if department is None:
        department = _extract_department(query, metadata) or _infer_recent_department(session_id)
    if preferred_time is None:
        preferred_time = extract_time_phrase(query)
        if not preferred_time:
            numeric_hour = _extract_numeric_hour(query)
            if numeric_hour is not None:
                preferred_time = f"{numeric_hour}시"
    if not date_hint:
        date_hint = _extract_date_phrase(query) or _extract_date_phrase(last_bot_answer)
    day_only = _extract_day_only(query)
    if day_only and not _extract_date_phrase(query):
        base_date = _parse_date_only(date_hint) if date_hint else timezone.localdate()
        adjusted = (
            _build_date_same_month(base_date, day_only)
            or _build_date_from_base_day(base_date, day_only)
        ) if base_date else None
        if adjusted:
            date_hint = f"{adjusted.month}월 {adjusted.day}일"
            if preferred_time and _extract_date_phrase(preferred_time):
                preferred_time = date_hint
            else:
                preferred_time = _merge_date_with_time(preferred_time, date_hint)
                if not preferred_time:
                    preferred_time = date_hint
    preferred_time = _merge_date_with_time(preferred_time, date_hint)
    asap = asap or contains_asap(query)
    preferred_time = normalize_preferred_time(preferred_time, asap)
    closed_reply = maybe_reject_closed_date(preferred_time or "")
    if closed_reply:
        return {"reply": closed_reply, "sources": []}

    if not department and not preferred_time:
        return {
            "reply": "예약을 위해 진료과와 희망 날짜/시간을 알려주세요.",
            "sources": [],
        }
    if not department:
        if preferred_time:
            return {"reply": "예약을 위해 진료과를 알려주세요.", "sources": []}
        return {"reply": "예약을 위해 진료과를 알려주세요.", "sources": []}
    if not preferred_time:
        return {
            "reply": f"{department} 진료로 도와드리겠습니다. 희망 날짜/시간을 알려주세요.",
            "sources": [],
        }
    if not has_specific_time(preferred_time):
        if not asap:
            return {
                "reply": f"{department} 진료로 도와드리겠습니다. {build_time_followup_message(preferred_time)}",
                "sources": [],
            }

    doctor_name = _extract_doctor_name(query, metadata)
    if not doctor_name:
        tool_context = build_tool_context(session_id, metadata)
        doctor_result = execute_tool("doctor_list", {"department": department}, tool_context)
        if isinstance(doctor_result, dict) and doctor_result.get("status") in {"not_found", "error"}:
            return {
                "reply": doctor_result.get("reply_text")
                or "해당 진료과 의료진 정보를 찾지 못했습니다. 원하시면 진료과명을 정확히 알려주세요.",
                "sources": [],
            }
        payload = {
            "reply": f"{department} 의료진을 선택해 주세요. 선택 후 예약을 진행합니다.",
            "sources": [],
        }
        if isinstance(doctor_result, dict) and doctor_result.get("table"):
            payload["table"] = doctor_result["table"]
        return payload

    tool_context = build_tool_context(session_id, metadata)
    result = execute_tool(
        "reservation_create",
        {
            "department": department,
            "preferred_time": preferred_time,
            "doctor_name": doctor_name,
        },
        tool_context,
    )
    if isinstance(result, dict):
        if result.get("reply_text"):
            return {"reply": result["reply_text"], "sources": []}
        if result.get("status") == "ok":
            return {
                "reply": (
                    f"{department} 진료 예약 요청이 접수되었습니다. "
                    f"희망 일정은 {preferred_time}입니다."
                ),
                "sources": [],
            }
        if result.get("status") == "error":
            return {
                "reply": "현재 예약을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                "sources": [],
            }
    return {
        "reply": "예약 처리 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        "sources": [],
    }





def _build_symptom_department_reply(query: str) -> str | None:
    q = (query or "").strip()
    if not q or not is_symptom_department_request(q):
        return None
    department = match_symptom_department(q)
    wants_booking = has_booking_intent(q) or has_symptom_time_booking_intent(q)
    guide_entry = match_symptom_guide(q)
    guide_department = guide_entry.get("department") if guide_entry else None
    guide_summary = guide_entry.get("summary") if guide_entry else ""
    guide_causes = guide_entry.get("possible_causes", []) if guide_entry else []
    if guide_department:
        department = guide_department
    # 암 관련 시 단계 구분 추가
    cancer_keywords = ["암", "유방암", "위암", "대장암", "간암", "폐암"]
    is_cancer_related = any(k in q for k in cancer_keywords)
    if guide_summary or guide_causes:
        cause_text = guide_summary or ", ".join(guide_causes[:3])
        if cause_text and department:
            if is_cancer_related:
                # 암 단계 구분: 초기 vs 진행성
                if any(k in q for k in ["초기", "조기", "일찍", "빨리"]):
                    cause_text += " 이는 초기 단계일 수 있으니 조기 진단을 권장합니다."
                elif any(k in q for k in ["출혈", "통증 심해", "생명", "위험", "응급"]):
                    cause_text += " 이는 진행성 단계일 수 있으니 즉시 응급실을 방문하세요."
                else:
                    cause_text += " 암 단계에 따라 초기(조기 발견) 또는 진행성(응급 필요)으로 구분될 수 있습니다."
            if wants_booking:
                return (
                    f"말씀하신 증상은 일반적으로 {cause_text}와 관련될 수 있습니다. "
                    f"정확한 판단은 진료를 통해 가능하니 {department} 진료를 권장드리며, "
                    "원하시면 희망 날짜/시간을 알려주세요."
                )
            return (
                f"말씀하신 증상은 일반적으로 {cause_text}와 관련될 수 있습니다. "
                f"정확한 판단은 진료를 통해 가능하니 {department} 진료를 권장드립니다."
            )
    if not department:
        if wants_booking:
            return (
                "증상에 따라 적절한 진료과가 달라질 수 있습니다. 주요 증상을 조금 더 알려주시면 "
                "맞는 진료과를 추천하고 예약까지 도와드리겠습니다."
            )
        return (
            "증상에 따라 적절한 진료과가 달라질 수 있습니다. 주요 증상을 조금 더 알려주시면 "
            "맞는 진료과를 추천해 드리겠습니다. 원하시면 예약도 도와드릴 수 있습니다."
        )
    if has_time_hint(q):
        if wants_booking:
            return (
                f"말씀하신 증상 기준으로는 {department} 진료가 필요할 수 있습니다. "
                "말씀하신 시간대로 예약을 진행해도 될까요? 예약자 이름과 연락처도 알려주세요."
            )
        return (
            f"말씀하신 증상 기준으로는 {department} 진료가 필요할 수 있습니다. "
            "해당 시간대로 예약을 원하시면 알려주세요."
        )
    if wants_booking:
        return (
            f"말씀하신 증상 기준으로는 {department} 진료가 필요할 수 있습니다. "
            "예약을 진행하려면 희망 날짜와 시간을 알려주세요."
        )
    return (
        f"말씀하신 증상 기준으로는 {department} 진료가 필요할 수 있습니다. "
        "원하시면 예약을 도와드리겠습니다. 희망 날짜/시간을 알려주세요."
    )


def _reservation_history_style(query: str) -> tuple[int, str, str]:
    q = (query or "").strip()
    if not q:
        return 0, "table", "예약 내역"
    if "다다음" in q:
        return 2, "single", "다다음 예약"
    if any(k in q for k in ["세번째", "세 번째", "3번째"]):
        return 2, "single", "세 번째 예약"
    if any(k in q for k in ["두번째", "두 번째", "2번째", "둘째"]):
        return 1, "single", "두 번째 예약"
    if "다음" in q:
        return 1, "single", "다음 예약"
    if any(k in q for k in ["내역", "목록", "리스트", "보여", "정리", "표", "일정"]):
        return 0, "table", "예약 내역"
    return 0, "table", "예약 내역"

# RAG core: called after cache check to build context and decide tool usage.
def run_rag(
    user_message: str,
    session_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> dict:
    """
    병원 안내용 RAG 파이프라인 진입점.

    1) 사용자 질문 임베딩
    2) FAISS 벡터 검색
    3) 상위 문서 텍스트(context)만 정제해서 LLM으로 전달
    4) Gemini 기반 답변 생성
    """
    try:
        settings = get_settings()

        query = (user_message or "").strip()
        request_id = metadata.get("request_id") if isinstance(metadata, dict) else ""
        logger.info("RAG start: request_id=%s query_len=%s", request_id, len(query))
        if not query:
            return {"reply": "질문이 비어 있습니다. 다시 입력해 주세요.", "sources": []}

        if is_smalltalk_query(query):
            return {"reply": DEFAULT_GREETING_REPLY, "sources": []}

        safety = build_safety_response(query)
        if safety:
            return {
                "reply": safety.reply,
                "sources": [{"type": "static", "name": "safety", "category": safety.category}],
            }

        medical_history = handle_medical_history(query, session_id, metadata)
        if medical_history:
            return medical_history

        followup = handle_reservation_followup(query, session_id, metadata)
        if followup:
            return followup

        if is_fixed_info_query(query):
            # 복합 의도(예: "위치 알려주고 예약해줘")인 경우 조기 리턴하지 않고 아래 메인 로직(RAG+Tool)으로 진행
            if not (has_booking_intent(query) or _is_doctor_query(query) or should_use_tools(query, metadata=metadata)):
                store = get_vector_store()
                all_meta = list(store._metadata.values())
                trusted_meta = _filter_trusted_info_meta(all_meta)
                trusted_contexts = [extract_context_text(meta) for meta in trusted_meta]
                trusted_contexts = [txt for txt in trusted_contexts if txt]
                static_answer = get_static_answer(query, trusted_contexts, trusted_meta)
                if static_answer:
                    return {
                        "reply": static_answer.reply,
                        "sources": static_answer.sources,
                    }

        if (
            _is_doctor_query(query)
            or is_doctor_followup(query, session_id)
            or is_doctor_department_followup(query, session_id)
        ) and not has_booking_intent(query) and not is_fixed_info_query(query):
            tool_context = build_tool_context(session_id, metadata)
            is_followup = is_doctor_followup(query, session_id) or is_doctor_department_followup(
                query, session_id
            )
            explicit_department = _extract_department(query, None)
            if not explicit_department and not is_followup and not _has_auth_context(metadata):
                return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
            if not explicit_department and not is_followup:
                meta_doctor = None
                if isinstance(metadata, dict):
                    meta_doctor = (
                        metadata.get("doctor_name")
                        or metadata.get("doctor")
                        or metadata.get("doctorName")
                    )
                if isinstance(meta_doctor, str) and meta_doctor.strip():
                    doctor_name = meta_doctor.strip()
                    display_name = _format_doctor_display_name(doctor_name, None)
                    reply_name = _format_doctor_reply_name(doctor_name)
                    payload = {
                        "reply": f"예약 의료진은 {reply_name}입니다.",
                        "sources": [],
                        "table": _build_doctor_table(
                            [{"name": display_name, "title": None, "phone": None}]
                        ),
                    }
                    return payload
                history = execute_tool(
                    "reservation_history",
                    {"offset": 0, "limit": 1, "reply_style": "single", "label": "예약"},
                    tool_context,
                )
                if isinstance(history, dict) and history.get("reservations"):
                    first = history["reservations"][0]
                    doctor_name = first.get("doctor_name")
                    if doctor_name:
                        display_name = _format_doctor_display_name(doctor_name, None)
                        reply_name = _format_doctor_reply_name(doctor_name)
                        payload = {
                            "reply": f"예약 의료진은 {reply_name}입니다.",
                            "sources": [],
                            "table": _build_doctor_table(
                                [{"name": display_name, "title": None, "phone": None}]
                            ),
                        }
                        return payload
                recent_doctor = infer_recent_doctor_name(session_id)
                if recent_doctor:
                    display_name = _format_doctor_display_name(recent_doctor, None)
                    reply_name = _format_doctor_reply_name(recent_doctor)
                    payload = {
                        "reply": f"예약 의료진은 {reply_name}입니다.",
                        "sources": [],
                        "table": _build_doctor_table(
                            [{"name": display_name, "title": None, "phone": None}]
                        ),
                    }
                    return payload
                return {
                    "reply": "예약 의료진 정보를 찾지 못했습니다. 진료과를 알려주시면 의료진을 안내해 드릴게요.",
                    "sources": [],
                }
            inferred_department = (
                explicit_department
                or _extract_department(query, metadata)
                or infer_recent_department(session_id)
            )
            if not inferred_department:
                slot_reply = build_slot_fill_response("doctor_list", query, tool_context)
                if slot_reply:
                    return {"reply": slot_reply, "sources": []}
            department = inferred_department
            result = execute_tool(
                "doctor_list",
                {"department": department} if department else {},
                tool_context,
            )
            if isinstance(result, dict) and result.get("reply_text"):
                payload = {"reply": result["reply_text"], "sources": []}
                if result.get("table"):
                    payload["table"] = result["table"]
                return payload
            return {
                "reply": "의료진 정보를 확인하는 데 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                "sources": [],
            }

        symptom_time_booking = has_symptom_time_booking_intent(query)
        if has_booking_intent(query) or symptom_time_booking:
            symptom_department = match_symptom_department(query)
            if not symptom_department:
                guide_entry = match_symptom_guide(query)
                symptom_department = guide_entry.get("department") if guide_entry else None
            if symptom_department and has_time_hint(query):
                if not _has_auth_context(metadata):
                    return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
                doctor_name = _extract_doctor_name(query, None)
                if not doctor_name:
                    tool_context = build_tool_context(session_id, metadata)
                    doctor_result = execute_tool(
                        "doctor_list",
                        {"department": symptom_department},
                        tool_context,
                    )
                    if isinstance(doctor_result, dict) and doctor_result.get("status") in {"not_found", "error"}:
                        return {
                            "reply": doctor_result.get("reply_text")
                            or "해당 진료과 의료진 정보를 찾지 못했습니다. 원하시면 진료과명을 정확히 알려주세요.",
                            "sources": [],
                        }
                    payload = {
                        "reply": f"{symptom_department} 의료진을 선택해 주세요. 선택 후 예약을 진행합니다.",
                        "sources": [],
                    }
                    if isinstance(doctor_result, dict) and doctor_result.get("table"):
                        payload["table"] = doctor_result["table"]
                    return payload
                asap = contains_asap(query)
                preferred_time = extract_time_phrase(query)
                preferred_time = normalize_preferred_time(preferred_time, asap)
                closed_reply = maybe_reject_closed_date(preferred_time or "")
                if closed_reply:
                    return {"reply": closed_reply, "sources": []}
                if not preferred_time:
                    return {
                        "reply": f"{symptom_department} 진료로 도와드리겠습니다. 희망 날짜/시간을 알려주세요.",
                        "sources": [],
                    }
                if not has_specific_time(preferred_time) and not asap:
                    return {
                        "reply": f"{symptom_department} 진료로 도와드리겠습니다. {build_time_followup_message(preferred_time)}",
                        "sources": [],
                    }
                tool_context = build_tool_context(session_id, metadata)
                result = execute_tool(
                    "reservation_create",
                    {
                        "department": symptom_department,
                        "preferred_time": preferred_time,
                        "doctor_name": doctor_name,
                    },
                    tool_context,
                )
                if isinstance(result, dict):
                    if result.get("reply_text"):
                        return {"reply": result["reply_text"], "sources": []}
                    if result.get("status") == "ok":
                        return {
                            "reply": (
                                f"{symptom_department} 진료 예약 요청이 접수되었습니다. "
                                f"희망 일정은 {preferred_time}입니다."
                            ),
                            "sources": [],
                        }
                    if result.get("status") == "error":
                        return {
                            "reply": "현재 예약을 처리하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
                            "sources": [],
                        }
                return {
                    "reply": "예약 처리 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                    "sources": [],
                }

        symptom_reply = _build_symptom_department_reply(query)
        has_booking_intent_val = has_booking_intent(query) or symptom_time_booking
        has_explicit_department = _extract_department(query, metadata) is not None
        has_symptom_department = (
            match_symptom_department(query) or match_symptom_guide(query)
        )
        has_time_hint_val = has_time_hint(query)
        if symptom_reply and not (
            has_booking_intent_val
            and (has_explicit_department or (has_symptom_department and has_time_hint_val))
        ):
            return {"reply": symptom_reply, "sources": []}

        early_tool = should_use_tools(query, metadata=metadata)
        if early_tool:
            tool_context = build_tool_context(session_id, metadata)
            tool_name = classify_tool_intent(query, metadata=metadata)
            if "예약" in query and has_reschedule_cue(query):
                tool_name = "reservation_reschedule"
            if tool_name == "reservation_reschedule" and not (
                has_reschedule_cue(query) or has_doctor_change_cue(query)
            ):
                tool_name = "reservation_create"
            if tool_name in {"reservation_history", "reservation_cancel", "reservation_reschedule"}:
                if not _has_auth_context(metadata):
                    return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
            if tool_name == "wait_status":
                inferred_department = _extract_department(query, metadata)
                if not inferred_department:
                    inferred_department = infer_wait_department(tool_context)
                slot_reply = build_slot_fill_response(tool_name, query, tool_context)
                if slot_reply and not inferred_department:
                    return {"reply": slot_reply, "sources": []}
                wait_args: Dict[str, Any] = {}
                if inferred_department:
                    wait_args["department"] = inferred_department
                result = execute_tool("wait_status", wait_args, tool_context)
                if isinstance(result, dict) and result.get("reply_text"):
                    return {"reply": result["reply_text"], "sources": []}
                if isinstance(result, dict) and result.get("status") == "not_found":
                    return {
                        "reply": "대기 현황을 확인하지 못했습니다. 진료과를 다시 확인해 주세요.",
                        "sources": [],
                    }
                if isinstance(result, dict) and result.get("status") == "error":
                    return {
                        "reply": "현재 시스템에서 대기 현황 확인이 어렵습니다. 잠시 후 다시 시도해 주세요.",
                        "sources": [],
                    }
            if tool_name == "reservation_create":
                if not _has_auth_context(metadata):
                    return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
                closed_reply = maybe_reject_closed_date(query)
                if closed_reply:
                    return {"reply": closed_reply, "sources": []}
                department = _extract_department(query, metadata) or _infer_recent_department(session_id)
                # 증상 기반 진료과 자동 추출
                if not department:
                    symptom_dept = match_symptom_department(query)
                    if not symptom_dept:
                        guide_entry = match_symptom_guide(query)
                        symptom_dept = guide_entry.get("department") if guide_entry else None
                    department = symptom_dept
                preferred_time = _extract_preferred_time(query, metadata)
                doctor_name = _extract_doctor_name(query, metadata)
                if doctor_name and not department:
                    return {
                        "reply": "어느 진료과 예약인지 알려주세요.",
                        "sources": [],
                    }
                if department and preferred_time and not has_specific_time(query):
                    return {"reply": TIME_REQUIRED_REPLY, "sources": []}
                if department and not preferred_time:
                    if doctor_name:
                        return {
                            "reply": f"{doctor_name} 의료진으로 예약을 진행합니다. 희망 날짜/시간을 알려주세요.",
                            "sources": [],
                        }
                    doctor_result = execute_tool(
                        "doctor_list",
                        {"department": department},
                        tool_context,
                    )
                    reply_text = (
                        f"{department} 의료진을 선택한 뒤 희망 날짜/시간을 알려주세요."
                    )
                    payload = {"reply": reply_text, "sources": []}
                    if isinstance(doctor_result, dict) and doctor_result.get("table"):
                        payload["table"] = doctor_result["table"]
                    return payload

        # 1) 질문 임베딩 생성
        embeddings = embed_texts([query])
        if not embeddings:
            raise ValueError("임베딩을 생성할 수 없습니다.")

        query_vector = np.array(embeddings[0], dtype="float32")

        # 2) FAISS 검색
        store = get_vector_store()
        top_k = getattr(settings, "top_k", 3)
        search_results = store.search(query_vector, top_k)
        all_meta = list(store._metadata.values())
        all_numbers = collect_contact_numbers(all_meta)

        def _general_fallback_reply() -> str:
            if all_numbers.get("대표번호"):
                return (
                    "죄송합니다. 관련된 정보를 찾지 못해 정확한 안내가 어렵습니다. "
                    f"자세한 사항은 병원 대표번호({all_numbers['대표번호']})로 문의해 주시기 바랍니다."
                )
            return (
                "죄송합니다. 관련된 정보를 찾지 못해 정확한 안내가 어렵습니다. "
                "병원 안내 데스크로 문의해 주시기 바랍니다."
            )

        if not search_results:
            return {
                "reply": _general_fallback_reply(),
                "sources": [],
            }

        # ✅ (중요) IndexFlatL2는 distance(작을수록 좋음). threshold로 거르지 말고 top_k 그대로 사용.
        # ✅ (중요) 간단 키워드 필터로 관련 문서 우선 선택
        relevant_results = _keyword_filter(query, search_results)
        logger.info(
            "RAG search: request_id=%s results=%s filtered=%s",
            request_id,
            len(search_results),
            len(relevant_results),
        )

        # 상위 몇 개만 컨텍스트로 사용
        max_docs = 3
        contexts_text: List[str] = []
        used_meta: List[Dict[str, Any]] = []
        used_scores: List[float] = []

        for dist, meta in relevant_results[:max_docs]:
            txt = extract_context_text(meta)
            if not txt:
                continue
            contexts_text.append(txt)
            used_meta.append(meta)
            used_scores.append(float(dist))

        parking_keys = _get_parking_keywords(query)
        if parking_keys:
            has_parking = any(
                any(k in ctx for k in parking_keys) for ctx in contexts_text
            )
            if not has_parking:
                fallback_contexts = []
                fallback_meta = []
                for meta in store._metadata.values():
                    txt = extract_context_text(meta)
                    if txt and any(k in txt for k in parking_keys):
                        fallback_contexts.append(txt)
                        fallback_meta.append(meta)
                        if len(fallback_contexts) >= max_docs:
                            break
                if fallback_contexts:
                    contexts_text = fallback_contexts
                    used_meta = fallback_meta
                    used_scores = [0.0] * len(fallback_meta)
                    logger.info("RAG keyword fallback: parking matches=%s", len(contexts_text))
            contexts_text = _strip_parking_settlement_info(contexts_text, query)

        static_answer = get_static_answer(query, contexts_text, all_meta)
        if static_answer:
            return {
                "reply": static_answer.reply,
                "sources": static_answer.sources,
            }
        doc_ids = [m.get("id") for m in used_meta if isinstance(m, dict) and m.get("id") is not None]
        logger.info("RAG contexts: request_id=%s used=%s doc_ids=%s", request_id, len(contexts_text), doc_ids)
        if not contexts_text:
            return {
                "reply": _general_fallback_reply(),
                "sources": [],
            }

        # 3) LLM 호출: meta 전체가 아니라 순수 텍스트 리스트만 전달
        use_tools = should_use_tools(query, metadata=metadata)
        tool_context = build_tool_context(session_id, metadata)
        tool_name = classify_tool_intent(query, metadata=metadata) if use_tools else None
        inferred_wait_department = None
        if use_tools and tool_name == "wait_status":
            inferred_wait_department = _extract_department(query, metadata)
            if not inferred_wait_department:
                inferred_wait_department = infer_wait_department(tool_context)
        slot_reply = build_slot_fill_response(tool_name, query, tool_context) if use_tools else None
        if slot_reply:
            if not (tool_name == "wait_status" and inferred_wait_department):
                return {
                    "reply": slot_reply,
                    "sources": [],
                }
        if use_tools and tool_name == "reservation_history":
            if not _has_auth_context(metadata):
                    return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
            offset, reply_style, label = _reservation_history_style(query)
            result = execute_tool(
                "reservation_history",
                {"offset": offset, "limit": 5, "reply_style": reply_style, "label": label},
                tool_context,
            )
            if isinstance(result, dict) and result.get("reply_text"):
                payload = {"reply": result["reply_text"], "sources": []}
                if result.get("table"):
                    payload["table"] = result["table"]
                return payload
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "예약 내역을 찾지 못했습니다. 원하시면 예약을 도와드리겠습니다. 진료과와 희망 날짜/시간을 알려주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 시스템에서 확인이 어렵습니다. 예약 번호나 연락처를 알려주시면 확인해 드리겠습니다.",
                    "sources": [],
                }
        if use_tools and tool_name == "wait_status":
            department = inferred_wait_department or _extract_department(query, metadata)
            wait_args: Dict[str, Any] = {}
            if department:
                wait_args["department"] = department
            result = execute_tool("wait_status", wait_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "대기 현황을 확인하지 못했습니다. 진료과를 다시 확인해 주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 시스템에서 대기 현황 확인이 어렵습니다. 잠시 후 다시 시도해 주세요.",
                    "sources": [],
                }
        if use_tools and tool_name == "reservation_cancel":
            cancel_args = {"cancel_all": True} if _has_bulk_cancel_cue(query) else {}
            cancel_args["cancel_text"] = query
            result = execute_tool("reservation_cancel", cancel_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "취소할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "현재 시스템에서 확인이 어렵습니다. 예약 번호나 연락처를 알려주시면 확인해 드리겠습니다.",
                    "sources": [],
                }
        if use_tools and "예약" in query and _has_reschedule_cue(query):
            tool_name = "reservation_reschedule"
        if use_tools and tool_name == "reservation_reschedule" and not (
            _has_reschedule_cue(query) or _has_doctor_change_cue(query)
        ):
            tool_name = "reservation_create"
        if use_tools and tool_name == "reservation_create":
            if not _has_auth_context(metadata):
                return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        if use_tools and tool_name == "reservation_reschedule":
            new_department = _extract_department(query, metadata)
            reschedule_args: Dict[str, Any] = {}
            if _has_time_or_date_hint(query) or any(
                marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
            ):
                reschedule_args["new_time"] = query
            if new_department:
                reschedule_args["new_department"] = new_department
            if not reschedule_args:
                return {
                    "reply": "변경할 날짜/시간이나 진료과를 알려주세요.",
                    "sources": [],
                }
            result = execute_tool("reservation_reschedule", reschedule_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
            if isinstance(result, dict) and result.get("status") == "not_found":
                return {
                    "reply": "변경할 예약을 찾지 못했습니다. 예약 번호나 연락처를 알려주세요.",
                    "sources": [],
                }
            if isinstance(result, dict) and result.get("status") == "error":
                return {
                    "reply": "변경할 날짜/시간이나 진료과를 다시 알려주세요.",
                    "sources": [],
                }
        sources_hash = _build_sources_hash(used_meta)
        # 암 관련 질문 시 프롬프트 튜닝: 초기/진행성 단계 구분 지시 추가
        if any(k in query for k in ["암", "유방암", "위암", "대장암", "간암", "폐암", "암센터"]):
            contexts_text.insert(0, "암 관련 증상 질문 시, 초기 단계(조기 발견 가능)와 진행성 단계(응급 필요)를 구분하여 응답하라. 예: 유방 멍울은 초기일 수 있으나, 출혈 동반 시 응급.")
        raw_reply = call_gemini_with_rag(
            query,
            contexts_text,
            tool_context=tool_context,
            use_tools=use_tools,
            sources_hash=sources_hash or None,
        )
        if not raw_reply:
            return {
                "reply": _general_fallback_reply(),
                "sources": [],
            }

        reply_text = clean_response(raw_reply)
        sources = [] if use_tools else _build_sources(used_meta, used_scores)
        return {"reply": reply_text, "sources": sources}

    except Exception as e:
        logger.exception("Error in RAG pipeline: %s", e)
        return {
            "reply": (
                "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다. "
                "잠시 후 다시 시도해 주시기 바랍니다."
            ),
            "sources": [],
        }


# Cache gate: called by API flow to reuse cached answers unless tool intent.
def run_rag_with_cache(
    user_message: str,
    session_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> dict:
    """
    1) DB(ChatCache)에서 동일 질문 캐시 조회
    2) 있으면 → 바로 리턴 (hit_count 증가)
    3) 없으면 → 기존 run_rag 실행 후 결과를 캐시에 저장
    """
    query = (user_message or "").strip()
    request_id = metadata.get("request_id") if isinstance(metadata, dict) else ""
    if not query:
        return {"reply": "질문이 비어 있습니다. 다시 입력해 주세요.", "sources": []}

    if needs_reservation_login_guard(query) and not _has_auth_context(metadata):
        return {"reply": AUTH_REQUIRED_REPLY, "sources": []}

    settings = get_settings()
    safety = build_safety_response(query)
    if safety:
        sources = [{"type": "static", "name": "safety", "category": safety.category}]
        save_cache_response(
            query=query,
            intent="safety",
            cache_scope=CACHE_SCOPE_QUERY_ONLY,
            rag_index_version=settings.rag_index_version,
            top_k=settings.top_k,
            prompt_version=settings.prompt_version,
            response=safety.reply,
            context_text="",
            context_hash="",
            sources_hash="",
            sources=sources,
        )
        logger.info("RAG safety: request_id=%s category=%s", request_id, safety.category)
        return {"reply": safety.reply, "sources": sources}

    medical_history = handle_medical_history(query, session_id, metadata)
    if medical_history:
        return medical_history

    followup = handle_reservation_followup(query, session_id, metadata)
    if followup:
        return followup

    if (
        _is_doctor_query(query)
        or is_doctor_followup(query, session_id)
        or is_doctor_department_followup(query, session_id)
    ) and not has_booking_intent(query):
        logger.info("RAG tool flow: request_id=%s cache bypass", request_id)
        return run_rag(query, session_id=session_id, metadata=metadata)

    if has_booking_intent(query) or has_symptom_time_booking_intent(query):
        logger.info("RAG tool flow: request_id=%s cache bypass", request_id)
        return run_rag(query, session_id=session_id, metadata=metadata)

    symptom_reply = _build_symptom_department_reply(query)
    has_booking_intent_val = has_booking_intent(query)
    has_explicit_department = _extract_department(query, metadata) is not None
    if symptom_reply and not (has_booking_intent_val and has_explicit_department):
        return {"reply": symptom_reply, "sources": []}

    use_tools = should_use_tools(query, metadata=metadata)
    if use_tools:
        tool_name = classify_tool_intent(query, metadata=metadata)
        if tool_name in {"reservation_history", "reservation_cancel", "reservation_reschedule"}:
            if not _has_auth_context(metadata):
                return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        if tool_name == "wait_status":
            tool_context = build_tool_context(session_id, metadata)
            inferred_department = _extract_department(query, metadata)
            if not inferred_department:
                inferred_department = infer_wait_department(tool_context)
            slot_reply = build_slot_fill_response(tool_name, query, tool_context)
            if slot_reply and not inferred_department:
                return {"reply": slot_reply, "sources": []}
            wait_args: Dict[str, Any] = {}
            if inferred_department:
                wait_args["department"] = inferred_department
            result = execute_tool("wait_status", wait_args, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
        logger.info("RAG tool flow: request_id=%s cache bypass", request_id)
        return run_rag(query, session_id=session_id, metadata=metadata)

    cached = get_cached_response(
        query=query,
        intent="rag",
        cache_scope=CACHE_SCOPE_QUERY_ONLY,
        rag_index_version=settings.rag_index_version,
        top_k=settings.top_k,
        prompt_version=settings.prompt_version,
    )
    if cached:
        logger.info("RAG cache hit: request_id=%s scope=%s", request_id, CACHE_SCOPE_QUERY_ONLY)
        return {"reply": cached.response, "sources": cached.sources or []}

    # 2) 캐시 없으면 → 원래 RAG 실행
    logger.info("RAG cache miss: request_id=%s scope=%s", request_id, CACHE_SCOPE_QUERY_ONLY)
    result = run_rag(query, session_id=session_id, metadata=metadata)
    reply_text = result.get("reply") or ""
    sources = result.get("sources") or []

    # 증상 추천 추가 (RAG 응답에 진료과 추천 통합)
    symptom_reply = _build_symptom_department_reply(query)
    if symptom_reply and not has_booking_intent(query):
        # RAG 응답이 이미 진료과 언급하지 않으면 추가
        if "진료과" not in reply_text and "외과" not in reply_text and "내과" not in reply_text:
            reply_text += f" {symptom_reply}"

    # 3) 캐시 저장
    if reply_text:
        save_cache_response(
            query=query,
            intent="rag",
            cache_scope=CACHE_SCOPE_QUERY_ONLY,
            rag_index_version=settings.rag_index_version,
            top_k=settings.top_k,
            prompt_version=settings.prompt_version,
            response=reply_text,
            context_text="",
            context_hash="",
            sources_hash="",
            sources=sources,
        )

    return result
