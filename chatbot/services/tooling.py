from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import date, datetime, timedelta, time as dt_time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from django.contrib.auth import get_user_model
from django.db import connections
from django.db.models import Q
from django.db.models.functions import Replace
from django.db.models import Value
from django.utils import timezone

from chatbot.config import get_settings
from chatbot.models import (
    ChatMessage,
    HospitalReservation,
    Notification,
    Reservation,
    ToolAuditLog,
    WaitStatus,
)

logger = logging.getLogger(__name__)

AUTH_REQUIRED_REPLY = "로그인 후 이용해 주세요, 전화 문의는 대표번호 1577-3330으로 부탁드립니다."
from chatbot.services.common import AUTH_METADATA_KEYS
DEPARTMENT_REQUIRED_REPLY = "예약을 위해 진료과명을 알려주세요."
TIME_REQUIRED_REPLY = "예약 희망 날짜와 시간을 알려주세요."

@dataclass
class ToolContext:
    session_id: str | None = None
    metadata: Dict[str, Any] | None = None
    request_id: str | None = None
    user_id: str | None = None


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


def _has_auth_context(context_or_metadata: ToolContext | Dict[str, Any] | None) -> bool:
    if not context_or_metadata:
        return False
    
    # If it's a ToolContext object
    if hasattr(context_or_metadata, "user_id"):
        if context_or_metadata.user_id:
            return True
        metadata = getattr(context_or_metadata, "metadata", {}) or {}
    # If it's a dictionary
    elif isinstance(context_or_metadata, dict):
        if context_or_metadata.get("user_id"):
            return True
        metadata = context_or_metadata
    else:
        return False

    return any(metadata.get(key) for key in AUTH_METADATA_KEYS)


def _metadata_intent_hint(metadata: Dict[str, Any] | None) -> Optional[bool]:
    if not metadata or not isinstance(metadata, dict):
        return None

    raw_use_tools = metadata.get("use_tools")
    if isinstance(raw_use_tools, bool):
        return raw_use_tools
    if isinstance(raw_use_tools, str):
        if raw_use_tools.strip().lower() in {"true", "1", "yes", "y"}:
            return True
        if raw_use_tools.strip().lower() in {"false", "0", "no", "n"}:
            return False

    raw_intent = metadata.get("intent") or metadata.get("route") or metadata.get("tool_intent")
    if isinstance(raw_intent, str):
        normalized = raw_intent.strip().lower()
        if normalized in {
            "tool",
            "tools",
            "reservation",
            "wait_status",
            "notification",
            "session",
            "doctor_list",
            "doctor",
        }:
            return True
        if normalized in {"rag", "static", "info", "faq", "knowledge"}:
            return False

    return None


def _matches_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


from chatbot.services.intents.keywords import (
    DATE_KOR_PATTERN,
    DATE_SLASH_PATTERN,
    DAY_ONLY_PATTERN,
    TIME_HINT_PATTERN,
)

DOCTOR_QUERY_KEYWORDS = [
    "의사",
    "의료진",
    "교수",
    "선생님",
    "닥터",
    "전문의",
    "진료의",
    "의료진 소개",
    "의료진 목록",
    "의료진 리스트",
    "의사 목록",
    "의사 리스트",
    "doctor",
    "physician",
    "admin",
]
ADMIN_EXCLUDE_USERNAMES = {"admin", "administrator", "root", "superuser"}
DOCTOR_TITLE_TOKENS = ["교수", "전문의", "의사", "선생님", "닥터", "doctor", "dr"]
DOCTOR_DISPLAY_SUFFIX_PATTERN = re.compile(r"\s*\(([^)]+)\)\s*$")
EXCLUDED_DOCTOR_DEPARTMENTS = {"admin", "원무과"}


def _is_doctor_query(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in DOCTOR_QUERY_KEYWORDS)


def _has_doctor_choice_cue(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(token in lowered for token in DOCTOR_TITLE_TOKENS)


def _metadata_tool_hint(metadata: Dict[str, Any] | None) -> Optional[str]:
    if not metadata or not isinstance(metadata, dict):
        return None
    raw_tool = metadata.get("tool_name") or metadata.get("tool") or metadata.get("tool_intent")
    if isinstance(raw_tool, str):
        normalized = raw_tool.strip().lower()
        if normalized in {
            "reservation_lookup",
            "reservation_create",
            "reservation_cancel",
            "reservation_history",
            "medical_history",
            "wait_status",
            "notification_send",
            "session_history",
            "doctor_list",
        }:
            return normalized
    return None


def _get_metadata_value(metadata: Dict[str, Any] | None, keys: List[str]) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _get_patient_id(metadata: Dict[str, Any] | None) -> str | None:
    return _get_metadata_value(
        metadata,
        [
            "patient_id",
            "patientId",
            "patient_identifier",
            "patientIdentifier",
            "patient_pk",
            "patientPk",
            "user_id",
            "account_id",
            "auth_user_id",
        ],
    )


@lru_cache(maxsize=1)
def _load_departments() -> set[str]:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    path = raw_dir / "departments.txt"
    departments: set[str] = set()
    if not path.exists():
        return departments
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.endswith("과") and len(text) <= 12:
            departments.add(text)
    return departments


@lru_cache(maxsize=256)
def _classify_tool_intent_with_llm(query: str) -> Optional[bool]:
    if not query:
        return None
    try:
        from chatbot.services.gemini_client import call_llm_with_failover
    except Exception as exc:  # pragma: no cover - import should generally succeed
        logger.warning("LLM intent classifier import failed: %s", exc)
        return None

    system_prompt = (
        "You are a router for a hospital chatbot. Decide whether the user request "
        "requires TOOL calls for live actions (reservation lookup/create/cancel/history, medical history, "
        "wait status, notification send, session history, doctor list) or can be answered from documents. "
        "If the user asks to check/change/cancel a reservation, medical history, wait status, send notifications, "
        "view session history, or list doctors -> TOOL. If the user asks about parking, location, hours, "
        "departments, admissions, costs, reservation guidance/policy, or general info -> RAG. "
        "Respond with only TOOL or RAG."
    )
    settings = get_settings()
    result = call_llm_with_failover(
        system_prompt,
        query,
        temperature=0.0,
        primary_override=settings.intent_llm_provider,
        model_override=settings.intent_llm_model,
    )
    if not result:
        return None
    token = re.split(r"\s+", result.strip().lower())[0].strip(".,:;!\"'")
    if token in {"tool", "tools"}:
        return True
    if token in {"rag", "static", "info"}:
        return False
    return None


def should_use_tools(user_message: str, metadata: Dict[str, Any] | None = None) -> bool:
    q = (user_message or "").strip()
    if not q:
        return False

    if _is_doctor_query(q):
        if any(token in q for token in ["예약", "접수", "대기", "순번"]):
            pass
        else:
            return True

    medical_history_keywords = [
        "진료내역",
        "진료 내역",
        "진료기록",
        "진료 기록",
        "진료이력",
        "진료 이력",
    ]
    if _matches_any(q, medical_history_keywords):
        return False


    metadata_hint = _metadata_intent_hint(metadata)
    if metadata_hint is not None:
        return metadata_hint

    q_lower = q.lower()

    tool_keywords = [
        "예약",
        "예약확인",
        "예약 확인",
        "예약조회",
        "예약 조회",
        "예약변경",
        "예약 변경",
        "예약취소",
        "예약 취소",
        "예약내역",
        "예약 내역",
        "예약이력",
        "예약 이력",
        "접수",
        "접수확인",
        "접수 확인",
        "접수조회",
        "접수 조회",
        "대기",
        "대기시간",
        "대기 시간",
        "대기현황",
        "대기 현황",
        "대기번호",
        "순번",
        "알림",
        "문자",
        "sms",
        "카카오",
        "카톡",
        "푸시",
        "이메일",
        "email",
        "메일",
        "히스토리",
        "history",
        "기록",
        "이전 대화",
        "세션",
        "session",
        "대화 기록",
    ]
    non_tool_keywords = [
        "주차",
        "parking",
        "주차요금",
        "주차 요금",
        "주차비",
        "위치",
        "주소",
        "오시는 길",
        "오시는길",
        "교통",
        "버스",
        "지하철",
        "예약 방법",
        "예약방법",
        "예약 안내",
        "예약안내",
        "예약 절차",
        "예약절차",
        "예약 가능",
        "예약가능",
        "예약 시간",
        "예약시간",
        "예약 문의",
        "예약문의",
        # "진료시간",
        # "진료 시간",
        "접수시간",
        "접수 시간",
        "운영시간",
        "운영 시간",
        "진료과",
        "진료 과",
        "진료과목",
        "입원",
        "퇴원",
        "병실",
        "비용",
        "요금",
        "진료비",
        "검사비",
        "연락처",
        "대표번호",
        "전화",
        "콜센터",
        "암센터",
        "진료내역",
        "진료 내역",
        "진료기록",
        "진료 기록",
        "진료이력",
        "진료 이력",
    ]
    ambiguous_cues = [
        "확인",
        "조회",
        "내역",
        "이력",
        "현황",
        "상태",
    ]
    reservation_existing_cues = [
        "예약확인",
        "예약 확인",
        "예약조회",
        "예약 조회",
        "예약내역",
        "예약 내역",
        "예약이력",
        "예약 이력",
        "예약기록",
        "예약 기록",
        "내 예약",
        "내예약",
        "예약시간",
        "예약 시간",
        "예약일정",
        "예약 일정",
        "예약스케줄",
        "예약 스케줄",
        "예약했",
        "예약 한",
        "예약한",
        "예약됨",
        "예약 된",
        "예약됐",
        "예약 됐",
        "예약되어",
        "예약되었",
        "예약있는지",
        "예약 있는지",
    ]

    tool_hit = _matches_any(q_lower, tool_keywords)
    non_tool_hit = _matches_any(q_lower, non_tool_keywords)

    if tool_hit and not non_tool_hit:
        return True
    if non_tool_hit and not tool_hit:
        return False

    if "예약" in q and _matches_any(q, reservation_existing_cues):
        return True

    ambiguous = tool_hit or non_tool_hit or _matches_any(q_lower, ambiguous_cues)
    if ambiguous:
        llm_decision = _classify_tool_intent_with_llm(q)
        if llm_decision is not None:
            logger.info("tool intent via llm: %s -> %s", q[:120], llm_decision)
            return llm_decision

    return False if non_tool_hit else tool_hit


@lru_cache(maxsize=256)
def _classify_tool_name_with_llm(query: str) -> Optional[str]:
    if not query:
        return None
    try:
        from chatbot.services.gemini_client import call_llm_with_failover
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM tool intent import failed: %s", exc)
        return None

    system_prompt = (
        "You are a router for a hospital chatbot. Choose the single best tool name "
        "for the user request. Return one of: reservation_lookup, reservation_create, "
        "reservation_cancel, reservation_reschedule, reservation_history, medical_history, wait_status, "
        "notification_send, session_history, doctor_list, unknown. Respond with only the tool name."
    )
    settings = get_settings()
    result = call_llm_with_failover(
        system_prompt,
        query,
        temperature=0.0,
        primary_override=settings.intent_llm_provider,
        model_override=settings.intent_llm_model,
    )
    if not result:
        return None
    token = re.split(r"\s+", result.strip().lower())[0].strip(".,:;!\"'")
    if token in {
        "reservation_lookup",
        "reservation_create",
        "reservation_cancel",
        "reservation_reschedule",
        "reservation_history",
        "medical_history",
        "wait_status",
        "notification_send",
        "session_history",
        "doctor_list",
    }:
        return token
    return None


def classify_tool_intent(user_message: str, metadata: Dict[str, Any] | None = None) -> Optional[str]:
    q = (user_message or "").strip().lower()
    if not q:
        return None

    meta_tool = _metadata_tool_hint(metadata)
    if meta_tool:
        return meta_tool

    cancel_keywords = ["예약취소", "예약 취소", "취소", "예약 취소해", "예약 취소하고"]
    reschedule_keywords = [
        "예약변경",
        "예약 변경",
        "예약 바꿔",
        "예약 바꿔줘",
        "예약 미뤄",
        "예약 미뤄줘",
        "예약 연기",
        "예약 연기해",
        "시간 변경",
        "시간 바꿔",
        "시간 미뤄",
        "진료과 변경",
        "진료과 바꿔",
        "진료과 바꿔줘",
        "부서 변경",
        "부서 바꿔",
        "과 변경",
        "과 바꿔",
    ]
    reschedule_generic_cues = ["변경", "바꿔", "옮겨", "미뤄", "연기"]
    medical_history_keywords = [
        "진료내역",
        "진료 내역",
        "진료기록",
        "진료 기록",
        "진료이력",
        "진료 이력",
    ]
    if _matches_any(q, medical_history_keywords):
        return "medical_history"

    history_keywords = [
        "예약내역",
        "예약 내역",
        "예약이력",
        "예약 이력",
        "예약 기록",
        "예약 기록 조회",
        "예약시간",
        "예약 시간",
        "예약일정",
        "예약 일정",
        "예약스케줄",
        "예약 스케줄",
        "다음 예약",
        "다음예약",
        "다음 일정",
        "다음 일정 알려",
        "다음 일정 보여",
        "다음 예약 알려",
        "가장 가까운 예약",
        "내 예약",
        "내예약",
    ]
    lookup_keywords = ["예약조회", "예약 조회", "예약확인", "예약 확인", "예약 확인해", "접수확인", "접수 확인"]
    wait_keywords = ["대기", "대기시간", "대기 시간", "순번", "대기현황", "대기 현황"]
    notification_keywords = ["알림", "문자", "sms", "카카오", "카톡", "이메일", "email", "푸시"]
    session_keywords = ["이전 대화", "대화 기록", "세션 기록", "세션 히스토리", "chat history", "session history"]
    create_keywords = [
        "예약해",
        "예약 해",
        "예약해줘",
        "예약 해줘",
        "예약해주세요",
        "예약 해주세요",
        "예약잡아",
        "예약 잡아",
        "예약잡아줘",
        "예약 잡아줘",
        "예약 추가",
        "추가 예약",
        "예약 추가해",
        "예약 추가해줘",
        "하나 더",
        "한 개 더",
        "한건 더",
        "한 건 더",
        "또 하나",
        "예약하고",
        "예약 하고",
        "예약하고싶",
        "예약 하고싶",
        "예약하고 싶",
        "예약 하고 싶",
        "예약잡고싶",
        "예약 잡고싶",
        "예약할",
        "예약 할",
        "예약하려",
        "예약 하려",
        "예약하려고",
        "예약 하려고",
        "예약할래",
        "예약 할래",
        "예약할게",
        "예약 할게",
        "예약 신청",
        "예약 접수",
        "예약 요청",
        "예약 원해",
        "예약 원함",
        "예약 부탁",
    ]
    existing_keywords = [
        "확인",
        "조회",
        "내역",
        "이력",
        "기록",
        "예약시간",
        "예약 시간",
        "예약일정",
        "예약 일정",
        "예약스케줄",
        "예약 스케줄",
        "예약했",
        "예약 한",
        "예약한",
        "예약됨",
        "예약 된",
        "예약됐",
        "예약 됐",
        "예약되어",
        "예약되었",
        "예약있는지",
        "예약 있는지",
        "예약 잡혀",
        "예약 잡혔",
        "예약 잡혀있",
        "예약 잡혀 있는",
        "내 예약",
        "내예약",
    ]

    if _matches_any(q, cancel_keywords):
        return "reservation_cancel"
    if _matches_any(q, reschedule_keywords):
        return "reservation_reschedule"
    if "예약" in q and any(cue in q for cue in reschedule_generic_cues):
        return "reservation_reschedule"
    if "예약" in q and any(
        token in q for token in ["변경", "바꿔", "바꾸", "미뤄", "연기", "옮겨", "조정", "수정"]
    ):
        return "reservation_reschedule"
    if _matches_any(q, history_keywords):
        return "reservation_history"
    if _matches_any(q, lookup_keywords):
        return "reservation_lookup"
    if _matches_any(q, wait_keywords):
        return "wait_status"
    if _matches_any(q, notification_keywords):
        return "notification_send"
    if _matches_any(q, session_keywords):
        return "session_history"

    has_existing_cue = "예약" in q and _matches_any(q, existing_keywords)
    has_create_cue = _matches_any(q, create_keywords)
    has_reservation_cue = (
        "예약" in q
        or _matches_any(q, cancel_keywords)
        or _matches_any(q, reschedule_keywords)
        or any(cue in q for cue in reschedule_generic_cues)
        or has_existing_cue
        or has_create_cue
    )
    department = _extract_department(user_message, metadata)
    preferred_time = _extract_preferred_time(user_message, metadata)

    if has_existing_cue and not has_create_cue:
        return "reservation_history"
    if "예약" in q and (has_create_cue or ((department or preferred_time) and not has_existing_cue)):
        return "reservation_create"
    if _is_doctor_query(q) and not has_reservation_cue and not _matches_any(q, wait_keywords):
        return "doctor_list"
    if "예약" in q:
        patient_identifier = _get_patient_id(metadata)
        phone = _extract_phone(user_message, metadata)
        if patient_identifier or phone:
            return "reservation_history"

    if should_use_tools(user_message, metadata=metadata):
        return _classify_tool_name_with_llm(user_message)
    return None


PHONE_PATTERN = re.compile(r"(?:0\d{1,2})[-\s]?\d{3,4}[-\s]?\d{4}")
RESERVATION_ID_PATTERN = re.compile(
    r"(?:예약번호|예약 번호|접수번호|접수 번호|예약 id|예약ID)\s*[:：]?\s*(\d{3,})",
    re.IGNORECASE,
)



def _normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) not in {10, 11}:
        return None
    return digits


def _extract_phone(text: str, metadata: Dict[str, Any] | None) -> str | None:
    meta_phone = _get_metadata_value(metadata, ["patient_phone", "phone", "tel"])
    if meta_phone:
        normalized = _normalize_phone(meta_phone)
        if normalized:
            return normalized
    if not text:
        return None
    match = PHONE_PATTERN.search(text)
    if not match:
        return None
    return _normalize_phone(match.group(0))


def _extract_reservation_id(text: str, metadata: Dict[str, Any] | None) -> int | None:
    raw = _get_metadata_value(metadata, ["reservation_id", "reservationId", "id"])
    if raw and raw.isdigit():
        return int(raw)
    if not text:
        return None
    match = RESERVATION_ID_PATTERN.search(text)
    if match and match.group(1).isdigit():
        return int(match.group(1))
    return None


def _extract_department(text: str, metadata: Dict[str, Any] | None) -> str | None:
    meta_dept = _get_metadata_value(metadata, ["department", "dept", "진료과"])
    if meta_dept:
        return _normalize_department(meta_dept)
    if not text:
        return None
    departments = _load_departments()
    compact_text = re.sub(r"\s+", "", text)
    for dept in sorted(departments, key=len, reverse=True):
        if dept in text:
            return dept
        if dept and re.sub(r"\s+", "", dept) in compact_text:
            return dept
    return None


def _extract_preferred_time(text: str, metadata: Dict[str, Any] | None) -> str | None:
    meta_time = _get_metadata_value(metadata, ["preferred_time", "preferredTime", "예약시간", "희망시간"])
    if meta_time:
        return meta_time
    if not text:
        return None
    if DATE_KOR_PATTERN.search(text) or DATE_SLASH_PATTERN.search(text) or DAY_ONLY_PATTERN.search(text):
        return "date"
    if any(k in text for k in ["오늘", "내일", "모레", "이번주", "다음주"]):
        return "relative"
    if TIME_HINT_PATTERN.search(text):
        return "time"
    if any(k in text.lower() for k in ["am", "pm"]):
        return "time"
    return None


def _extract_doctor_name(text: str, metadata: Dict[str, Any] | None) -> str | None:
    meta_name = _get_metadata_value(metadata, ["doctor_name", "doctor", "doctorName"])
    if meta_name:
        return meta_name
    if not text:
        return None
    match = DOCTOR_NAME_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def _infer_department_from_session(session_id: str | None) -> str | None:
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


def _extract_channel(text: str, metadata: Dict[str, Any] | None) -> str | None:
    meta_channel = _get_metadata_value(metadata, ["channel", "알림채널"])
    if meta_channel:
        return meta_channel
    q = (text or "").lower()
    if "sms" in q or "문자" in q:
        return "sms"
    if "카카오" in q or "카톡" in q or "kakao" in q:
        return "kakao"
    if "이메일" in q or "email" in q or "메일" in q:
        return "email"
    if "푸시" in q or "push" in q:
        return "push"
    return None


def build_slot_fill_response(
    tool_name: str | None,
    user_message: str,
    context: ToolContext | None,
) -> Optional[str]:
    if not tool_name:
        return None
    text = user_message or ""
    metadata = context.metadata if context else None

    if tool_name == "doctor_list":
        department = _extract_department(text, metadata)
        if not department:
            return "어느 진료과 의료진을 찾으시나요?"

    if tool_name in {"reservation_lookup", "reservation_cancel"}:
        reservation_id = _extract_reservation_id(text, metadata)
        phone = _extract_phone(text, metadata)
        patient_id = _get_patient_id(metadata)
        if not reservation_id and not phone and not patient_id:
            return "예약 확인을 위해 예약 번호, 환자 ID 또는 연락처를 알려주세요."
    elif tool_name == "reservation_reschedule":
        reservation_id = _extract_reservation_id(text, metadata)
        phone = _extract_phone(text, metadata)
        patient_id = _get_patient_id(metadata)
        department = _extract_department(text, metadata)
        time_hint = _extract_preferred_time(text, metadata)
        doctor_name = _extract_doctor_name(text, metadata)
        if not reservation_id and not phone and not patient_id:
            return "예약 변경을 위해 예약 번호, 환자 ID 또는 연락처를 알려주세요."
        if not time_hint and not department:
            return "예약 변경을 위해 변경할 날짜/시간이나 진료과를 알려주세요."
        if _has_doctor_choice_cue(text) and not doctor_name:
            return "변경할 의료진 이름을 알려주세요. 없으면 '지정 없음'이라고 말씀해 주세요."
    elif tool_name == "reservation_history":
        phone = _extract_phone(text, metadata)
        patient_id = _get_patient_id(metadata)
        if not phone and not patient_id and not (context and context.session_id):
            return "예약 이력 조회를 위해 환자 ID 또는 연락처를 알려주세요."
    elif tool_name == "medical_history":
        phone = _extract_phone(text, metadata)
        patient_id = _get_patient_id(metadata)
        if not phone and not patient_id:
            return "진료내역을 확인하려면 환자 ID나 연락처를 알려주세요."
    elif tool_name == "reservation_create":
        department = _extract_department(text, metadata)
        preferred_time = _extract_preferred_time(text, metadata)
        if not department and not preferred_time:
            return "예약을 위해 진료과와 희망 날짜/시간을 알려주세요."
        if not department:
            return "예약을 위해 진료과를 알려주세요."
        if not preferred_time:
            return "예약을 위해 희망 날짜/시간을 알려주세요."
        doctor_name = _extract_doctor_name(text, metadata)
        if _has_doctor_choice_cue(text) and not doctor_name:
            return "원하시는 의료진 이름을 알려주세요. 없으면 '지정 없음'이라고 말씀해 주세요."
    elif tool_name == "wait_status":
        department = _extract_department(text, metadata)
        patient_id = _get_patient_id(metadata)
        phone = _extract_phone(text, metadata)
        if not department and not patient_id and not phone:
            return "대기 현황을 확인할 진료과를 알려주세요."
    elif tool_name == "notification_send":
        channel = _extract_channel(text, metadata)
        message = _get_metadata_value(metadata, ["message", "알림내용"])
        if not channel and not message:
            return "알림 채널(SMS/카카오/이메일)과 메시지 내용을 알려주세요."
        if not channel:
            return "알림 채널(SMS/카카오/이메일)을 알려주세요."
        if not message:
            return "보낼 메시지 내용을 알려주세요."
    elif tool_name == "session_history":
        if not (context and context.session_id):
            return "세션 기록 조회를 위해 session_id를 알려주세요."
    return None


# Tool registry: provides JSON schemas for LLM tool-calling.
def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "reservation_lookup",
                "description": "Look up a reservation by ID, patient ID, or phone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {"type": "integer"},
                        "patient_id": {"type": "string"},
                        "patient_phone": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reservation_create",
                "description": "Create a new reservation request. Include doctor_name or doctor_id when the user specifies a doctor.",
                "parameters": {
                    "type": "object",
                    "required": ["department", "preferred_time"],
                    "properties": {
                        "patient_name": {"type": "string"},
                        "patient_phone": {"type": "string"},
                        "department": {"type": "string"},
                        "preferred_time": {"type": "string"},
                        "doctor_name": {"type": "string"},
                        "doctor_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "channel": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reservation_cancel",
                "description": "Cancel a reservation by ID or patient phone. Use cancel_all to cancel all upcoming reservations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {"type": "integer"},
                        "patient_id": {"type": "string"},
                        "patient_phone": {"type": "string"},
                        "reason": {"type": "string"},
                        "cancel_all": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reservation_reschedule",
                "description": "Reschedule or change the department for an existing reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {"type": "integer"},
                        "patient_id": {"type": "string"},
                        "patient_phone": {"type": "string"},
                        "new_time": {"type": "string"},
                        "new_department": {"type": "string"},
                        "doctor_name": {"type": "string"},
                        "doctor_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reservation_history",
                "description": "List recent reservations for a patient or session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "patient_phone": {"type": "string"},
                        "session_id": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                        "reply_style": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "medical_history",
                "description": "List recent medical records for a patient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "patient_phone": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wait_status",
                "description": "Get current wait status for a department or patient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string"},
                        "patient_id": {"type": "string"},
                        "patient_phone": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doctor_list",
                "description": "List doctors for a department.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "notification_send",
                "description": "Create a notification request (SMS/Kakao/Email/Push).",
                "parameters": {
                    "type": "object",
                    "required": ["channel", "message"],
                    "properties": {
                        "channel": {"type": "string"},
                        "target": {"type": "string"},
                        "message": {"type": "string"},
                        "schedule_at": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "session_history",
                "description": "Fetch recent chat history by session_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            },
        },
    ]


ALLOWED_NOTIFICATION_CHANNELS = {
    "sms": "sms",
    "문자": "sms",
    "kakao": "kakao",
    "카카오": "kakao",
    "카톡": "kakao",
    "email": "email",
    "이메일": "email",
    "메일": "email",
    "push": "push",
    "푸시": "push",
}

SENSITIVE_TOOLS = {
    "reservation_lookup",
    "reservation_create",
    "reservation_cancel",
    "reservation_reschedule",
    "reservation_history",
    "medical_history",
    "session_history",
    "notification_send",
}


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _normalize_channel(value: str | None) -> str | None:
    if not value:
        return None
    normalized = ALLOWED_NOTIFICATION_CHANNELS.get(value.strip().lower())
    if normalized:
        return normalized
    return ALLOWED_NOTIFICATION_CHANNELS.get(value.strip())


def _normalize_department(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    departments = _load_departments()
    if not departments:
        return text
    normalized = re.sub(r"\s+", "", text)
    if text in departments:
        return text
    for dept in sorted(departments, key=len, reverse=True):
        if not dept:
            continue
        if dept in text:
            return dept
        if re.sub(r"\s+", "", dept) == normalized:
            return dept
        if normalized and re.sub(r"\s+", "", dept) in normalized:
            return dept
    return text


def _validate_department(value: str | None) -> bool:
    if not value:
        return False
    departments = _load_departments()
    if not departments:
        return True
    return value in departments


def _is_tool_authorized(name: str, context: ToolContext | None) -> bool:
    settings = get_settings()
    if not getattr(settings, "tool_auth_required", False):
        return True
    if name not in SENSITIVE_TOOLS:
        return True
    if not context:
        return False
    metadata = context.metadata or {}
    if context.user_id:
        return True
    if metadata.get("verified_user") or metadata.get("auth_user_id") or metadata.get("user_id"):
        return True
    return False


def _mask_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 7:
        return value
    return f"{digits[:3]}****{digits[-4:]}"


def _mask_args(args: Dict[str, Any]) -> Dict[str, Any]:
    masked = {}
    for key, value in (args or {}).items():
        if value is None:
            continue
        if key in {"patient_phone", "phone", "target"} and isinstance(value, str):
            masked[key] = _mask_phone(value)
        elif key in {"message", "reason"} and isinstance(value, str):
            masked[key] = value[:40]
        else:
            masked[key] = value
    return masked


def _format_schedule_text(value: str | None) -> tuple[str, str]:
    if not value:
        return "-", "-"
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if timezone.is_aware(dt):
            dt = dt.replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except ValueError:
        resolved = _resolve_requested_datetime(raw)
        if resolved:
            resolved = timezone.localtime(resolved)
            return resolved.strftime("%Y-%m-%d"), resolved.strftime("%H:%M")
        return raw, "-"


def _resolve_requested_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = _parse_preferred_datetime(value)
    if parsed:
        if timezone.is_naive(parsed):
            return timezone.make_aware(parsed, timezone.get_current_timezone())
        return parsed
    if any(word in value for word in ASAP_TIME_WORDS):
        return _build_asap_datetime()
    return None


def _extract_cancel_dates(text: str | None) -> list[date]:
    if not text:
        return []
    base = timezone.localdate()
    dates: list[date] = []
    month_candidates: list[tuple[int, int | None]] = []
    for match in DATE_KOR_PATTERN.finditer(text):
        year_text, month_text, day_text = match.group(1), match.group(2), match.group(3)
        try:
            year = int(year_text) if year_text else base.year
            month = int(month_text)
            day = int(day_text)
            dates.append(date(year, month, day))
            month_candidates.append((month, year_text and year or None))
        except (TypeError, ValueError):
            continue
    for match in DATE_SLASH_PATTERN.finditer(text):
        year_text, month_text, day_text = match.group(1), match.group(2), match.group(3)
        try:
            year = int(year_text) if year_text else base.year
            month = int(month_text)
            day = int(day_text)
            dates.append(date(year, month, day))
            month_candidates.append((month, year_text and year or None))
        except (TypeError, ValueError):
            continue
    for match in DATE_DASH_PATTERN.finditer(text):
        year_text, month_text, day_text = match.group(1), match.group(2), match.group(3)
        try:
            dates.append(date(int(year_text), int(month_text), int(day_text)))
        except (TypeError, ValueError):
            continue
    day_matches = DAY_ONLY_PATTERN.findall(text)
    if day_matches:
        month = base.month
        year = base.year
        explicit_months = {m for m, _ in month_candidates}
        explicit_years = {y for _, y in month_candidates if y}
        if len(explicit_months) == 1:
            month = next(iter(explicit_months))
        if len(explicit_years) == 1:
            year = next(iter(explicit_years))
        for day_text in day_matches:
            try:
                day = int(day_text)
            except (TypeError, ValueError):
                continue
            try:
                dates.append(date(year, month, day))
            except ValueError:
                continue
    unique: list[date] = []
    seen: set[date] = set()
    for value in dates:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _build_reservation_table(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "현재 예약 내역이 없습니다. 원하시면 예약을 도와드리겠습니다. 진료과와 희망 날짜/시간을 알려주세요."
    lines = ["예약 내역을 아래에 정리해 드리겠습니다.", "날짜 | 시간 | 과 | 담당의"]
    for item in items:
        raw_time = item.get("scheduled_at") or item.get("requested_time") or ""
        date_text, time_text = _format_schedule_text(raw_time)
        department = item.get("department") or "-"
        raw_doctor = item.get("doctor_name") or ""
        doctor_name = (
            _format_doctor_reply_name(raw_doctor) if raw_doctor else "의료진 미지정"
        )
        lines.append(f"{date_text} | {time_text} | {department} | {doctor_name}")
    return "\n".join(lines)


def _build_reservation_table_data(items: List[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    headers = ["날짜", "시간", "과", "담당의", "메모"]
    rows: List[List[str]] = []
    for item in items:
        raw_time = item.get("scheduled_at") or item.get("requested_time") or ""
        date_text, time_text = _format_schedule_text(raw_time)
        department = item.get("department") or "-"
        raw_doctor = item.get("doctor_name") or ""
        doctor_display = (
            _format_doctor_display_name(raw_doctor, item.get("doctor_code"))
            if raw_doctor
            else "의료진 미지정"
        )
        memo = (item.get("memo") or item.get("reason") or "").strip()
        rows.append([date_text, time_text, department, doctor_display, memo or "-"])
    return {"headers": headers, "rows": rows}


def _build_medical_history_table_data(
    records: List[Dict[str, Any]],
    include_notes: bool = True,
) -> Dict[str, List[List[str]]]:
    headers = ["날짜", "시간", "과", "상태"]
    if include_notes:
        headers.append("메모")
    rows: List[List[str]] = []
    for record in records:
        raw_time = record.get("reception_start_time") or ""
        date_text, time_text = _format_schedule_text(raw_time)
        department = record.get("department") or "-"
        status = record.get("status") or "-"
        row = [date_text, time_text, department, status]
        if include_notes:
            notes = (record.get("notes") or "").strip()
            row.append(notes or "-")
        rows.append(row)
    return {"headers": headers, "rows": rows}


def _build_medical_history_table(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "현재 진료내역이 없습니다. 필요하시면 예약을 도와드리겠습니다."
    table = _build_medical_history_table_data(records, include_notes=False)
    lines = [
        "진료 내역을 아래에 정리해 드리겠습니다.",
        "날짜 | 시간 | 과 | 상태",
    ]
    for row in table["rows"]:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _format_reservation_single(item: Dict[str, Any], label: str) -> str:
    raw_time = item.get("scheduled_at") or item.get("requested_time") or ""
    date_text, time_text = _format_schedule_text(raw_time)
    department = item.get("department") or "-"
    raw_doctor = item.get("doctor_name") or ""
    doctor_text = (
        f" {_format_doctor_reply_name(raw_doctor)} 의료진"
        if raw_doctor
        else ""
    )
    normalized_label = label.strip() if isinstance(label, str) and label.strip() else "예약"
    return (
        f"{normalized_label}은 {date_text} {time_text} {department}{doctor_text} 진료예요. "
        "변경이나 취소가 필요하면 말씀해 주세요."
    )


def _record_audit_log(
    name: str,
    context: ToolContext | None,
    args: Dict[str, Any],
    result: Dict[str, Any],
    latency_ms: int,
) -> None:
    try:
        ToolAuditLog.objects.create(
            request_id=(context.request_id if context else "") or "",
            session_id=(context.session_id if context else "") or "",
            user_id=(context.user_id if context else "") or "",
            tool_name=name,
            status=result.get("status") or "error",
            error_code=result.get("error_code") or result.get("message") or "",
            latency_ms=latency_ms,
            metadata={"args": _mask_args(args)},
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("audit log save failed: %s", exc)


def _merge_args_from_context(name: str, args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    if not context or not context.metadata:
        if name in {"session_history", "reservation_history"} and context and context.session_id:
            args = dict(args)
            args.setdefault("session_id", context.session_id)
        return args

    merged = dict(args or {})
    metadata = context.metadata

    if "patient_phone" not in merged:
        meta_phone = _get_metadata_value(metadata, ["patient_phone", "phone", "tel"])
        if meta_phone:
            merged["patient_phone"] = meta_phone

    if "patient_id" not in merged:
        meta_patient_id = _get_patient_id(metadata)
        if meta_patient_id:
            merged["patient_id"] = meta_patient_id
        elif context and context.user_id:
            merged["patient_id"] = context.user_id

    if name == "doctor_list":
        if "department" not in merged:
            meta_dept = _get_metadata_value(
                metadata,
                ["department", "dept", "진료과", "last_department", "recent_department"],
            )
            if meta_dept:
                merged["department"] = meta_dept

    if name == "reservation_create":
        if "department" not in merged:
            meta_dept = _get_metadata_value(metadata, ["department", "dept", "진료과"])
            if meta_dept:
                merged["department"] = meta_dept
        if "preferred_time" not in merged:
            meta_time = _get_metadata_value(metadata, ["preferred_time", "예약시간", "희망시간"])
            if meta_time:
                merged["preferred_time"] = meta_time
        if "patient_name" not in merged:
            meta_name = _get_metadata_value(metadata, ["patient_name", "name", "이름"])
            if meta_name:
                merged["patient_name"] = meta_name

    if name == "notification_send":
        if "channel" not in merged:
            meta_channel = _get_metadata_value(metadata, ["channel", "알림채널"])
            if meta_channel:
                merged["channel"] = meta_channel
        if "message" not in merged:
            meta_message = _get_metadata_value(metadata, ["message", "알림내용"])
            if meta_message:
                merged["message"] = meta_message

    if name in {"session_history", "reservation_history"}:
        if "session_id" not in merged and context.session_id:
            merged["session_id"] = context.session_id

    return merged


# Tool executor: called by tool loop to run server-side actions.
def execute_tool(name: str, args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    handlers = {
        "reservation_lookup": _reservation_lookup,
        "reservation_create": _reservation_create,
        "reservation_cancel": _reservation_cancel,
        "reservation_reschedule": _reservation_reschedule,
        "reservation_history": _reservation_history,
        "medical_history": _medical_history,
        "wait_status": _wait_status,
        "doctor_list": _doctor_list,
        "notification_send": _notification_send,
        "session_history": _session_history,
    }
    handler = handlers.get(name)
    if not handler:
        return {"status": "error", "message": f"Unknown tool: {name}"}

    if not _is_tool_authorized(name, context):
        result = {"status": "error", "error_code": "auth_required", "message": "authentication required"}
        _record_audit_log(name, context, args or {}, result, latency_ms=0)
        return result

    merged_args = _merge_args_from_context(name, args or {}, context)
    start_time = time.monotonic()
    try:
        result = handler(merged_args, context)
    except Exception as exc:  # pragma: no cover
        logger.exception("tool handler failed: %s", exc)
        result = {"status": "error", "error_code": "tool_exception", "message": "tool execution failed"}
    latency_ms = int((time.monotonic() - start_time) * 1000)
    _record_audit_log(name, context, merged_args, result, latency_ms=latency_ms)
    return result


def _get_hospital_reservations_qs() -> Any | None:
    try:
        from django.conf import settings

        if "hospital" not in settings.DATABASES:
            return None
        return HospitalReservation.objects.using("hospital")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital db unavailable: %s", exc)
        return None


def _get_hospital_reservations_qs_for_alias(alias: str) -> Any | None:
    try:
        return HospitalReservation.objects.using(alias)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital db unavailable for alias %s: %s", alias, exc)
        return None


def _get_hospital_patient_info(patient_identifier: str) -> Dict[str, Any] | None:
    if not patient_identifier:
        return None
    try:
        with connections["hospital"].cursor() as cursor:
            cursor.execute(
                "SELECT id, name, gender, age FROM patients_patient WHERE patient_id = %s LIMIT 1",
                [patient_identifier],
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "gender": row[2], "age": row[3]}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital patient lookup failed: %s", exc)
        return None


def _get_hospital_patient_user(patient_identifier: str) -> Dict[str, Any] | None:
    if not patient_identifier:
        return None
    try:
        with connections["hospital"].cursor() as cursor:
            cursor.execute(
                "SELECT id, name, phone FROM patient_user WHERE patient_id = %s LIMIT 1",
                [patient_identifier],
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "phone": row[2]}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital patient user lookup failed: %s", exc)
        return None


def _lookup_patient_identifier_by_phone(patient_phone: str | None) -> str | None:
    normalized = _normalize_phone(patient_phone)
    if not normalized:
        return None
    try:
        with connections["hospital"].cursor() as cursor:
            cursor.execute(
                "SELECT patient_id FROM patient_user WHERE REPLACE(phone, '-', '') = %s LIMIT 1",
                [normalized],
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            cursor.execute(
                "SELECT patient_id FROM patients_patient WHERE REPLACE(phone, '-', '') = %s LIMIT 1",
                [normalized],
            )
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital patient lookup by phone failed: %s", exc)
    return None


def _ensure_hospital_patient_record(
    patient_identifier: str,
    patient_name: str,
    patient_phone: str,
) -> Dict[str, Any] | None:
    info = _get_hospital_patient_info(patient_identifier)
    if info:
        return info
    user = _get_hospital_patient_user(patient_identifier)
    resolved_name = (patient_name or "").strip() or (user.get("name") if user else "") or "미상"
    resolved_phone = (patient_phone or "").strip() or (user.get("phone") if user else "") or ""
    user_account_id = user.get("id") if user else None
    now = timezone.localtime(timezone.now())
    try:
        with connections["hospital"].cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO patients_patient (
                    patient_id, name, phone, address, emergency_contact,
                    medical_history, allergies, created_at, updated_at, user_account_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    patient_identifier,
                    resolved_name,
                    resolved_phone,
                    "",
                    "",
                    "",
                    "",
                    now,
                    now,
                    user_account_id,
                ],
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital patient insert failed: %s", exc)
    return _get_hospital_patient_info(patient_identifier)


def _build_doctor_name(user: Any) -> str:
    last_name = getattr(user, "last_name", "") or ""
    first_name = getattr(user, "first_name", "") or ""
    name = f"{last_name}{first_name}".strip()
    if name:
        return name
    username = getattr(user, "username", "")
    return username or "담당의"


def _split_doctor_display(name: str | None) -> tuple[str, str | None]:
    if not name:
        return "", None
    text = str(name).strip()
    match = DOCTOR_DISPLAY_SUFFIX_PATTERN.search(text)
    if not match:
        return text, None
    base = text[: match.start()].strip()
    suffix = match.group(1).strip()
    return base, suffix


def _format_doctor_display_name(name: str | None, doctor_id: str | None) -> str:
    base, suffix = _split_doctor_display(name)
    if suffix:
        return f"{base} ({suffix})".strip()
    base = base.strip() if base else "의료진"
    if doctor_id:
        code = str(doctor_id).strip()
        if code and not code.isdigit():
            return f"{base} ({code})"
    return f"{base} (의료진)"


def _format_doctor_reply_name(name: str | None) -> str:
    base, _ = _split_doctor_display(name)
    base = base.strip()
    return base or "의료진"


def _normalize_doctor_name(name: str | None) -> str:
    if not name:
        return ""
    base, _ = _split_doctor_display(str(name))
    cleaned = re.sub(r"\s+", "", base)
    cleaned = re.sub(r"(교수|전문의|의사|선생님|닥터|doctor|dr\.?)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _get_default_doctor_info(department: str | None) -> Dict[str, Any] | None:
    try:
        User = get_user_model()
        qs = User.objects.using("hospital").all()
        doctor = None
        doctor_id_env = os.getenv("HOSPITAL_DEFAULT_DOCTOR_ID", "").strip()
        doctor_username_env = os.getenv("HOSPITAL_DEFAULT_DOCTOR_USERNAME", "").strip()
        if doctor_username_env:
            doctor = qs.filter(username=doctor_username_env).first()
        if not doctor and doctor_id_env.isdigit():
            doctor = qs.filter(id=int(doctor_id_env)).first()
        if not doctor and department:
            doctor = qs.filter(username__icontains=department).first()
        if not doctor:
            doctor = qs.filter(username="doctor").first()
        if not doctor:
            doctor = qs.filter(is_staff=True).first()
        if not doctor:
            doctor = qs.first()
        if not doctor:
            return None
        doctor_id = int(doctor.id)
        doctor_code = os.getenv("HOSPITAL_DEFAULT_DOCTOR_CODE", "").strip()
        if not doctor_code:
            doctor_code = f"D{timezone.localdate().year}{doctor_id:03d}"
        return {
            "doctor_id": doctor_id,
            "doctor_code": doctor_code,
            "doctor_username": getattr(doctor, "username", "") or str(doctor_id),
            "doctor_name": _build_doctor_name(doctor),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital doctor lookup failed: %s", exc)
        return None


DATE_KOR_PATTERN = re.compile(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일")
DATE_SLASH_PATTERN = re.compile(r"(?:(\d{4})\s*/\s*)?(\d{1,2})\s*/\s*(\d{1,2})")
DATE_DASH_PATTERN = re.compile(r"(\d{4})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})")
DAY_ONLY_PATTERN = re.compile(r"(?:^|\s)(\d{1,2})\s*일")
TIME_COLON_PATTERN = re.compile(r"(\d{1,2})\s*:\s*(\d{2})")
TIME_KOR_PATTERN = re.compile(
    r"(오전|오후|저녁|밤|새벽)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?"
)
ASAP_TIME_WORDS = ["당장", "가능한 빠른", "최대한 빨리", "빠른 시일", "가장 빠른", "급해", "긴급", "바로"]
RELATIVE_DAY_WORDS = {"오늘": 0, "금일": 0, "내일": 1, "모레": 2, "이번주": 0, "다음주": 7}
SAME_TIME_MARKERS = ["같은 시간", "동일하게", "시간은 동일", "시간은 그대로", "시간 동일"]
DOCTOR_NAME_PATTERN = re.compile(r"([가-힣]{2,4})\s*(?:교수|전문의|의사|선생님)")
WEEKDAY_WORDS = {
    "월요일": 0,
    "화요일": 1,
    "수요일": 2,
    "목요일": 3,
    "금요일": 4,
    "토요일": 5,
    "일요일": 6,
    "월요": 0,
    "화요": 1,
    "수요": 2,
    "목요": 3,
    "금요": 4,
    "토요": 5,
    "일요": 6,
}

CLINIC_WEEKDAY_START = dt_time(8, 30)
CLINIC_WEEKDAY_END = dt_time(17, 0)
CLINIC_SATURDAY_START = dt_time(8, 30)
CLINIC_SATURDAY_END = dt_time(12, 0)
CLINIC_SATURDAY_WEEKS = {1, 3}
CLINIC_CLOSED_REPLY = (
    "진료 예약 가능 시간이 아닙니다. 평일 08:30~17:00, "
    "토요일(1,3주) 08:30~12:00(공휴일 제외) 시간으로 알려주세요."
)


@lru_cache(maxsize=48)
def _fetch_holiday_dates(year: int, month: int) -> set[int]:
    settings = get_settings()
    api_key = (settings.holiday_api_key or "").strip()
    if not api_key:
        return set()
    params = {
        "serviceKey": api_key,
        "solYear": str(year),
        "solMonth": f"{month:02d}",
        "numOfRows": "100",
        "_type": "json",
    }
    try:
        response = requests.get(
            settings.holiday_api_base_url,
            params=params,
            timeout=settings.holiday_api_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("holiday api fetch failed: %s", exc)
        return set()
    items = (
        payload.get("response", {})
        .get("body", {})
        .get("items", {})
        .get("item", [])
    )
    if isinstance(items, dict):
        items = [items]
    dates: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        is_holiday = item.get("isHoliday") if "isHoliday" in item else item.get("is_holiday")
        if isinstance(is_holiday, str):
            if is_holiday.strip().upper() != "Y":
                continue
        elif isinstance(is_holiday, bool):
            if not is_holiday:
                continue
        elif isinstance(is_holiday, int):
            if is_holiday != 1:
                continue
        else:
            continue
        locdate = item.get("locdate") or item.get("locDate")
        try:
            if locdate is None:
                continue
            dates.add(int(locdate))
        except (TypeError, ValueError):
            continue
    return dates


def _is_holiday_date(value: date) -> bool:
    key = value.year * 10000 + value.month * 100 + value.day
    return key in _fetch_holiday_dates(value.year, value.month)


def _week_of_month(value: date) -> int:
    return (value.day - 1) // 7 + 1


def _is_closed_clinic_date(value: date) -> bool:
    if _is_holiday_date(value):
        return True
    weekday = value.weekday()
    if weekday == 6:
        return True
    if weekday == 5:
        return _week_of_month(value) not in CLINIC_SATURDAY_WEEKS
    return False


def _is_clinic_open_datetime(value: datetime) -> bool:
    local_dt = (
        timezone.localtime(value)
        if timezone.is_aware(value)
        else timezone.make_aware(value, timezone.get_current_timezone())
    )
    local_date = local_dt.date()
    if _is_holiday_date(local_date):
        return False
    weekday = local_date.weekday()
    if weekday == 6:
        return False
    local_time = local_dt.time()
    if weekday == 5:
        if _week_of_month(local_date) not in CLINIC_SATURDAY_WEEKS:
            return False
        return CLINIC_SATURDAY_START <= local_time <= CLINIC_SATURDAY_END
    return CLINIC_WEEKDAY_START <= local_time <= CLINIC_WEEKDAY_END


def _parse_preferred_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    iso = _parse_optional_datetime(text)
    if iso:
        return iso
    base_date = timezone.localdate()
    for word, delta_days in RELATIVE_DAY_WORDS.items():
        if word in text:
            base_date = base_date + timedelta(days=delta_days)
            break
    match = DATE_KOR_PATTERN.search(text)
    if not match:
        match = DATE_SLASH_PATTERN.search(text)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        year_int = int(year) if year else base_date.year
        candidate_date = base_date.replace(year=year_int, month=int(month), day=int(day))
        if not year:
            today = timezone.localdate()
            if candidate_date < today:
                candidate_date = candidate_date.replace(year=candidate_date.year + 1)
        base_date = candidate_date
    else:
        weekday_index = None
        for token, idx in WEEKDAY_WORDS.items():
            if token in text:
                weekday_index = idx
                break
        if weekday_index is not None:
            base_for_week = base_date
            if "다음주" in text:
                base_for_week = base_for_week + timedelta(days=7)
            delta = (weekday_index - base_for_week.weekday()) % 7
            candidate_date = base_for_week + timedelta(days=delta)
            if candidate_date < base_date:
                candidate_date = candidate_date + timedelta(days=7)
            base_date = candidate_date
    time_match = TIME_COLON_PATTERN.search(text)
    hour = None
    minute = None
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
    else:
        time_match = TIME_KOR_PATTERN.search(text)
        if time_match:
            period = time_match.group(1) or ""
            hour = int(time_match.group(2))
            minute = int(time_match.group(3) or 0)
            if period in {"오후", "저녁", "밤"} and hour < 12:
                hour += 12
            if period == "오전" and hour == 12:
                hour = 0
            if period == "새벽" and hour == 12:
                hour = 0
    if hour is None:
        return None
    local_dt = datetime(
        year=base_date.year,
        month=base_date.month,
        day=base_date.day,
        hour=hour,
        minute=minute or 0,
    )
    if timezone.is_naive(local_dt):
        return timezone.make_aware(local_dt, timezone.get_current_timezone())
    now_local = timezone.localtime(timezone.now())
    if local_dt < now_local and any(token in text for token in WEEKDAY_WORDS):
        if not any(token in text for token in ["오늘", "금일"]):
            local_dt = local_dt + timedelta(days=7)
    return local_dt


def _build_asap_datetime() -> datetime:
    return timezone.localtime(timezone.now()) + timedelta(hours=1)


def _parse_time_components(text: str) -> tuple[int | None, int | None]:
    match = TIME_COLON_PATTERN.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = TIME_KOR_PATTERN.search(text)
    if not match:
        return None, None
    period = match.group(1) or ""
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    if period in {"오후", "저녁", "밤"} and hour < 12:
        hour += 12
    if period == "오전" and hour == 12:
        hour = 0
    if period == "새벽" and hour == 12:
        hour = 0
    return hour, minute


def _has_time_component(text: str) -> bool:
    if not text:
        return False
    if TIME_COLON_PATTERN.search(text) or TIME_KOR_PATTERN.search(text):
        return True
    if any(marker in text for marker in SAME_TIME_MARKERS):
        return True
    if any(word in text for word in ASAP_TIME_WORDS):
        return True
    return False


def _has_date_component(text: str) -> bool:
    if not text:
        return False
    if DATE_KOR_PATTERN.search(text) or DATE_SLASH_PATTERN.search(text):
        return True
    if DAY_ONLY_PATTERN.search(text):
        return True
    if any(word in text for word in RELATIVE_DAY_WORDS):
        return True
    if any(word in text for word in WEEKDAY_WORDS):
        return True
    return False


def _build_rescheduled_datetime(text: str, base_dt: datetime) -> datetime | None:
    if not text:
        return None
    text = text.strip()
    keep_time = any(marker in text for marker in SAME_TIME_MARKERS)
    base_local = timezone.localtime(base_dt) if timezone.is_aware(base_dt) else base_dt
    year = base_local.year
    month = base_local.month
    day = base_local.day
    hour = base_local.hour
    minute = base_local.minute

    date_found = False
    time_found = False

    match = DATE_KOR_PATTERN.search(text) or DATE_SLASH_PATTERN.search(text)
    has_explicit_year = False
    if match:
        year_text = match.group(1)
        month_text = match.group(2)
        day_text = match.group(3)
        has_explicit_year = bool(year_text)
        year = int(year_text) if year_text else year
        month = int(month_text)
        day = int(day_text)
        date_found = True
    else:
        day_match = DAY_ONLY_PATTERN.search(text)
        if day_match:
            day = int(day_match.group(1))
            date_found = True

    parsed_hour, parsed_minute = _parse_time_components(text)
    if parsed_hour is not None:
        hour = parsed_hour
        minute = parsed_minute or 0
        time_found = True
    elif keep_time:
        time_found = True

    if not date_found and not time_found:
        return None

    new_dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
    if match and not has_explicit_year:
        today = timezone.localdate()
        if new_dt.date() < today:
            new_dt = new_dt.replace(year=new_dt.year + 1)
    if timezone.is_naive(new_dt):
        return timezone.make_aware(new_dt, timezone.get_current_timezone())
    return new_dt


def _create_hospital_appointment(
    patient_identifier: str | None,
    department: str,
    preferred_time: str,
    patient_name: str,
    patient_phone: str,
    reason: str,
    doctor_info: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if not patient_identifier:
        return None
    doctor_info = doctor_info or _get_default_doctor_info(department)
    if not doctor_info:
        return None
    patient_info = _ensure_hospital_patient_record(
        patient_identifier=patient_identifier,
        patient_name=patient_name,
        patient_phone=patient_phone,
    )
    resolved_name = patient_name or (patient_info.get("name") if patient_info else "") or "미상"
    resolved_gender = (patient_info.get("gender") if patient_info else "") or "U"
    resolved_age = patient_info.get("age") if patient_info else None
    start_time = _parse_preferred_datetime(preferred_time)
    if not start_time:
        if any(word in preferred_time for word in ASAP_TIME_WORDS):
            start_time = _build_asap_datetime()
        else:
            start_time = _build_asap_datetime()
    end_time = start_time + timedelta(minutes=30)
    now = timezone.localtime(timezone.now())
    appointment_id = uuid.uuid4().hex
    title = f"{department} 진료 예약"
    memo = reason or "AI 상담 예약 요청"
    with connections["hospital"].cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO patients_appointment (
                id, title, type, start_time, end_time,
                patient_identifier, patient_name, patient_gender, patient_age,
                doctor_code, doctor_username, doctor_name, doctor_department,
                status, memo, created_at, updated_at, created_by_id,
                doctor_id, patient_id
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s
            )
            """,
            [
                appointment_id,
                title,
                "예약",
                start_time,
                end_time,
                patient_identifier,
                resolved_name,
                resolved_gender,
                resolved_age,
                doctor_info["doctor_code"],
                doctor_info["doctor_username"],
                doctor_info["doctor_name"],
                department,
                "scheduled",
                memo,
                now,
                now,
                None,
                doctor_info["doctor_id"],
                patient_info.get("id") if patient_info else None,
            ],
        )
    return {"id": appointment_id, "start_time": start_time, "end_time": end_time}


def _hospital_reservation_payload(reservation: HospitalReservation) -> Dict[str, Any]:
    scheduled_at = reservation.start_time.isoformat() if reservation.start_time else ""
    end_time = reservation.end_time.isoformat() if reservation.end_time else ""
    return {
        "id": reservation.id,
        "title": reservation.title,
        "patient_identifier": reservation.patient_identifier,
        "patient_name": reservation.patient_name,
        "doctor_name": reservation.doctor_name,
        "department": reservation.doctor_department,
        "requested_time": scheduled_at,
        "scheduled_at": scheduled_at,
        "end_time": end_time,
        "status": reservation.status,
        "memo": (reservation.memo or "").strip(),
        "source": "hospital_db",
    }


def _reservation_lookup(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    if not _has_auth_context(context):
        return {
            "status": "auth_required",
            "message": "auth required",
            "reply_text": AUTH_REQUIRED_REPLY,
        }
    raw_reservation_id = args.get("reservation_id")
    reservation_id = _parse_int(raw_reservation_id)
    hospital_reservation_id = raw_reservation_id.strip() if isinstance(raw_reservation_id, str) else None
    patient_phone_raw = args.get("patient_phone")
    patient_phone = _normalize_phone(patient_phone_raw)
    patient_identifier = args.get("patient_identifier") or args.get("patient_id")

    if patient_phone_raw and not patient_phone:
        if not (patient_identifier or hospital_reservation_id or reservation_id):
            return {"status": "error", "message": "invalid patient_phone"}
        patient_phone_raw = None

    if not patient_identifier and (patient_phone_raw or patient_phone):
        patient_identifier = _lookup_patient_identifier_by_phone(patient_phone_raw or patient_phone)

    hospital_qs = _get_hospital_reservations_qs()
    if hospital_qs is not None and (hospital_reservation_id or patient_identifier):
        try:
            record = None
            if hospital_reservation_id:
                record = hospital_qs.filter(id=hospital_reservation_id).first()
            elif patient_identifier:
                base_qs = hospital_qs.filter(patient_identifier=patient_identifier).exclude(
                    status__iexact="cancelled"
                )
                now = timezone.now()
                record = (
                    base_qs.filter(start_time__gte=now).order_by("start_time").first()
                    or base_qs.order_by("-start_time").first()
                )
            if record:
                return {"status": "ok", "reservation": _hospital_reservation_payload(record)}
        except Exception as exc:  # pragma: no cover
            logger.warning("hospital reservation lookup failed: %s", exc)

    if patient_identifier and not reservation_id and not patient_phone:
        return {"status": "not_found"}

    qs = Reservation.objects.all()
    if reservation_id:
        qs = qs.filter(id=reservation_id)
    elif patient_phone:
        qs = qs.filter(patient_phone__in={patient_phone_raw or "", patient_phone})
    else:
        return {"status": "error", "message": "reservation_id, patient_id, or patient_phone required"}

    reservation = qs.order_by("-created_at").first()
    if not reservation:
        return {"status": "not_found"}

    return {"status": "ok", "reservation": _reservation_payload(reservation)}


def _reservation_create(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    department = _normalize_department(args.get("department"))
    preferred_time = args.get("preferred_time")
    patient_phone_raw = args.get("patient_phone")
    patient_phone = _normalize_phone(patient_phone_raw)
    patient_identifier = args.get("patient_identifier") or args.get("patient_id") or ""
    doctor_name = args.get("doctor_name") or args.get("doctor") or args.get("doctorName")
    doctor_id = args.get("doctor_id") or args.get("doctorId") or args.get("doctor_code")
    if context and context.metadata:
        if not doctor_name:
            doctor_name = _get_metadata_value(
                context.metadata, ["doctor_name", "doctor", "doctorName"]
            )
        if not doctor_id:
            doctor_id = _get_metadata_value(
                context.metadata, ["doctor_id", "doctorId", "doctor_code"]
            )
    if isinstance(doctor_name, str):
        normalized_name = doctor_name.strip()
        if any(token in normalized_name for token in ["지정 없음", "무관", "아무", "상관없"]):
            doctor_name = None
    if isinstance(doctor_id, str):
        normalized_id = doctor_id.strip()
        if any(token in normalized_id for token in ["없", "무관"]):
            doctor_id = None

    if not _has_auth_context(context):
        return {
            "status": "auth_required",
            "message": "auth required",
            "reply_text": AUTH_REQUIRED_REPLY,
        }

    if not department and context and context.session_id:
        department = _infer_department_from_session(context.session_id)

    if not department:
        return {"status": "ok", "message": "department required", "reply_text": DEPARTMENT_REQUIRED_REPLY}
    if not preferred_time:
        return {"status": "ok", "message": "preferred_time required", "reply_text": TIME_REQUIRED_REPLY}
    if _has_date_component(str(preferred_time)) and not _has_time_component(str(preferred_time)):
        return {"status": "ok", "message": "preferred_time required", "reply_text": TIME_REQUIRED_REPLY}
    if not _validate_department(department):
        return {"status": "ok", "message": "invalid department", "reply_text": DEPARTMENT_REQUIRED_REPLY}
    if patient_phone_raw and not patient_phone:
        if not patient_identifier:
            return {"status": "error", "message": "invalid patient_phone"}
        patient_phone_raw = None

    requested_dt = _resolve_requested_datetime(preferred_time)
    if requested_dt:
        now = timezone.localtime(timezone.now())
        if requested_dt < now:
            return {
                "status": "error",
                "message": "past datetime",
                "reply_text": "지난 날짜나 시간으로는 예약을 잡을 수 없습니다. 오늘 이후의 날짜와 시간을 알려주세요.",
            }
        if not _is_clinic_open_datetime(requested_dt):
            return {
                "status": "error",
                "message": "closed_hours",
                "reply_text": CLINIC_CLOSED_REPLY,
            }

    reservation = Reservation.objects.create(
        session_id=(context.session_id if context else "") or "",
        patient_name=args.get("patient_name") or "",
        patient_phone=patient_phone or (patient_phone_raw or ""),
        department=department,
        reason=args.get("reason") or "",
        requested_time_text=preferred_time,
        channel=args.get("channel") or "",
        status="pending",
    )
    hospital_payload = None
    doctor_info = None
    if doctor_name or doctor_id:
        doctor_info = _resolve_doctor_info(department, doctor_name, doctor_id)
        if not doctor_info:
            return {
                "status": "not_found",
                "message": "doctor not found",
                "reply_text": "요청하신 의료진 정보를 찾지 못했습니다. 의료진 이름을 다시 알려주세요.",
            }
    if patient_identifier:
        try:
            hospital_payload = _create_hospital_appointment(
                patient_identifier=str(patient_identifier),
                department=department,
                preferred_time=str(preferred_time),
                patient_name=args.get("patient_name") or "",
                patient_phone=patient_phone or (patient_phone_raw or ""),
                reason=args.get("reason") or "",
                doctor_info=doctor_info,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("hospital appointment create failed: %s", exc)
    if doctor_info:
        logger.info(
            "reservation create doctor: department=%s doctor_code=%s doctor_name=%s",
            department,
            doctor_info.get("doctor_code"),
            doctor_info.get("doctor_name"),
        )
    doctor_label = doctor_info.get("doctor_name") if doctor_info else None
    doctor_display = (
        _format_doctor_display_name(doctor_label, doctor_info.get("doctor_code"))
        if doctor_label
        else None
    )
    doctor_reply_name = _format_doctor_reply_name(doctor_label) if doctor_label else ""
    doctor_suffix = f"{doctor_reply_name} 의료진으로 " if doctor_reply_name else ""
    reservation_payload = _reservation_payload(reservation)
    if doctor_label:
        reservation_payload["doctor_name"] = doctor_label
    reservation_table = _build_reservation_table_data([reservation_payload])
    return {
        "status": "ok",
        "reservation": reservation_payload,
        "hospital_appointment": hospital_payload,
        "table": reservation_table,
        "reply_text": (
            f"{department} {doctor_suffix}진료 예약 요청이 접수되었습니다. 희망 일정은 {preferred_time}입니다. "
            "변경이나 취소가 필요하면 말씀해 주세요."
        ),
    }


def _reservation_cancel(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    if not _has_auth_context(context):
        return {
            "status": "auth_required",
            "message": "auth required",
            "reply_text": AUTH_REQUIRED_REPLY,
        }
    reservation_id = _parse_int(args.get("reservation_id"))
    raw_reservation_id = args.get("reservation_id")
    hospital_reservation_id = raw_reservation_id.strip() if isinstance(raw_reservation_id, str) else None
    patient_phone_raw = args.get("patient_phone")
    patient_phone = _normalize_phone(patient_phone_raw)
    patient_identifier = args.get("patient_identifier") or args.get("patient_id")
    raw_cancel_all = args.get("cancel_all") or args.get("cancelAll") or args.get("bulk")
    cancel_all = _parse_bool(raw_cancel_all)
    session_id = args.get("session_id") or (context.session_id if context else None)
    cancel_text = (args.get("cancel_text") or args.get("query") or args.get("text") or "").strip()
    cancel_dates = _extract_cancel_dates(cancel_text)

    def _filter_by_cancel_dates(records: list[Any]) -> list[Any]:
        if not cancel_dates:
            return records
        date_set = set(cancel_dates)
        filtered: list[Any] = []
        for record in records:
            record_dt = getattr(record, "start_time", None)
            if record_dt:
                record_dt = timezone.localtime(record_dt)
                if record_dt.date() in date_set:
                    filtered.append(record)
        return filtered

    if patient_phone_raw and not patient_phone:
        if not (patient_identifier or hospital_reservation_id or reservation_id):
            return {"status": "error", "message": "invalid patient_phone"}
        patient_phone_raw = None

    if not patient_identifier and (patient_phone_raw or patient_phone):
        patient_identifier = _lookup_patient_identifier_by_phone(patient_phone_raw or patient_phone)

    hospital_qs = _get_hospital_reservations_qs()
    cancel_by_dates = bool(cancel_dates)
    if hospital_qs is not None and (hospital_reservation_id or patient_identifier):
        try:
            now = timezone.now()
            if (cancel_all or cancel_by_dates) and patient_identifier:
                records = list(
                    hospital_qs.filter(
                        patient_identifier=patient_identifier,
                        start_time__gte=now,
                    )
                    .exclude(status__iexact="cancelled")
                    .order_by("start_time")
                )
                if cancel_by_dates:
                    records = _filter_by_cancel_dates(records)
                if records:
                    now_local = timezone.localtime(timezone.now())
                    with connections["hospital"].cursor() as cursor:
                        for record in records:
                            cursor.execute(
                                "UPDATE patients_appointment SET status = %s, updated_at = %s WHERE id = %s",
                                ["cancelled", now_local, record.id],
                            )
                    payloads = []
                    for record in records:
                        payload = _hospital_reservation_payload(record)
                        payload["status"] = "cancelled"
                        payloads.append(payload)
                    reply_text = (
                        f"\uCD1D {len(payloads)}\uAC74\uC758 \uC608\uC57D\uC744 \uCDE8\uC18C\uD588\uC2B5\uB2C8\uB2E4. "
                        "\uD544\uC694\uD558\uC2DC\uBA74 \uB2E4\uC2DC \uC608\uC57D\uC744 \uB3C4\uC640\uB4DC\uB9B4\uAC8C\uC694."
                    )
                    return {"status": "ok", "reservations": payloads, "reply_text": reply_text}
                reply_text = "해당 날짜 예약을 찾지 못했습니다." if cancel_by_dates else None
                if reply_text:
                    return {"status": "not_found", "reply_text": reply_text}
                return {"status": "not_found"}
            record = None
            if hospital_reservation_id:
                record = hospital_qs.filter(id=hospital_reservation_id).first()
            elif patient_identifier:
                records = list(
                    hospital_qs.filter(
                        patient_identifier=patient_identifier,
                        start_time__gte=now,
                    )
                    .exclude(status__iexact="cancelled")
                    .order_by("start_time")
                )
                if cancel_by_dates:
                    records = _filter_by_cancel_dates(records)
                record = records[0] if records else None
            if record:
                now = timezone.localtime(timezone.now())
                with connections["hospital"].cursor() as cursor:
                    cursor.execute(
                        "UPDATE patients_appointment SET status = %s, updated_at = %s WHERE id = %s",
                        ["cancelled", now, record.id],
                    )
                payload = _hospital_reservation_payload(record)
                payload["status"] = "cancelled"
                date_text, time_text = _format_schedule_text(payload.get("scheduled_at"))
                reply_text = (
                    f"{date_text} {time_text} {payload.get('department') or ''} 예약을 취소했습니다. "
                    "필요하시면 다시 예약을 도와드리겠습니다."
                ).strip()
                return {"status": "ok", "reservation": payload, "reply_text": reply_text}
        except Exception as exc:  # pragma: no cover
            logger.warning("hospital reservation cancel failed: %s", exc)
    qs = Reservation.objects.all()
    if reservation_id and not cancel_all:
        qs = qs.filter(id=reservation_id)
    elif patient_phone:
        qs = qs.filter(patient_phone__in={patient_phone_raw or "", patient_phone})
    elif session_id:
        qs = qs.filter(session_id=session_id)
    else:
        return {"status": "error", "message": "reservation_id or patient_phone required"}

    if cancel_all:
        qs = qs.exclude(status="cancelled")
        records = list(qs.order_by("-created_at"))
        now = timezone.now()
        cancellable: list[Reservation] = []
        for record in records:
            scheduled_at = _resolve_requested_datetime(record.requested_time_text)
            if scheduled_at and scheduled_at < now:
                continue
            if cancel_by_dates and scheduled_at:
                if scheduled_at.date() not in set(cancel_dates):
                    continue
            cancellable.append(record)
        if not cancellable:
            reply_text = "해당 날짜 예약을 찾지 못했습니다." if cancel_by_dates else None
            if reply_text:
                return {"status": "not_found", "reply_text": reply_text}
            return {"status": "not_found"}
        cancel_reason = args.get("reason") or ""
        for record in cancellable:
            record.status = "cancelled"
            record.cancel_reason = cancel_reason
            record.cancelled_at = now
            record.save(update_fields=["status", "cancel_reason", "cancelled_at", "updated_at"])
        reply_text = (
            f"\uCD1D {len(cancellable)}\uAC74\uC758 \uC608\uC57D\uC744 \uCDE8\uC18C\uD588\uC2B5\uB2C8\uB2E4. "
            "\uD544\uC694\uD558\uC2DC\uBA74 \uB2E4\uC2DC \uC608\uC57D\uC744 \uB3C4\uC640\uB4DC\uB9B4\uAC8C\uC694."
        )
        return {
            "status": "ok",
            "reservations": [_reservation_payload(record) for record in cancellable],
            "reply_text": reply_text,
        }

    if cancel_by_dates:
        records = list(qs.order_by("-created_at"))
        now = timezone.now()
        cancellable: list[Reservation] = []
        for record in records:
            scheduled_at = _resolve_requested_datetime(record.requested_time_text)
            if scheduled_at and scheduled_at < now:
                continue
            if scheduled_at and scheduled_at.date() not in set(cancel_dates):
                continue
            cancellable.append(record)
        if not cancellable:
            return {"status": "not_found", "reply_text": "해당 날짜 예약을 찾지 못했습니다."}
        cancel_reason = args.get("reason") or ""
        for record in cancellable:
            record.status = "cancelled"
            record.cancel_reason = cancel_reason
            record.cancelled_at = now
            record.save(update_fields=["status", "cancel_reason", "cancelled_at", "updated_at"])
        reply_text = (
            f"총 {len(cancellable)}건의 예약을 취소했습니다. "
            "필요하시면 다시 예약을 도와드릴게요."
        )
        return {
            "status": "ok",
            "reservations": [_reservation_payload(record) for record in cancellable],
            "reply_text": reply_text,
        }

    reservation = qs.order_by("-created_at").first()
    if not reservation:
        return {"status": "not_found"}

    reservation.status = "cancelled"
    reservation.cancel_reason = args.get("reason") or ""
    reservation.cancelled_at = timezone.now()
    reservation.save(update_fields=["status", "cancel_reason", "cancelled_at", "updated_at"])
    return {"status": "ok", "reservation": _reservation_payload(reservation)}


def _reservation_reschedule(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    if not _has_auth_context(context):
        return {
            "status": "auth_required",
            "message": "auth required",
            "reply_text": AUTH_REQUIRED_REPLY,
        }
    raw_reservation_id = args.get("reservation_id")
    hospital_reservation_id = raw_reservation_id.strip() if isinstance(raw_reservation_id, str) else None
    reservation_id = _parse_int(raw_reservation_id)
    patient_phone_raw = args.get("patient_phone")
    patient_phone = _normalize_phone(patient_phone_raw)
    patient_identifier = args.get("patient_identifier") or args.get("patient_id")
    new_time_text = (args.get("new_time") or args.get("preferred_time") or "").strip()
    new_department = _normalize_department(args.get("new_department") or args.get("department"))
    doctor_name = args.get("doctor_name") or args.get("doctor") or args.get("doctorName")
    doctor_id = args.get("doctor_id") or args.get("doctorId") or args.get("doctor_code")
    if context and context.metadata:
        if not doctor_name:
            doctor_name = _get_metadata_value(
                context.metadata, ["doctor_name", "doctor", "doctorName"]
            )
        if not doctor_id:
            doctor_id = _get_metadata_value(
                context.metadata, ["doctor_id", "doctorId", "doctor_code"]
            )
    if isinstance(doctor_name, str):
        normalized_name = doctor_name.strip()
        if any(token in normalized_name for token in ["지정 없음", "무관", "아무", "상관없"]):
            doctor_name = None
    if isinstance(doctor_id, str):
        normalized_id = doctor_id.strip()
        if any(token in normalized_id for token in ["없", "무관"]):
            doctor_id = None
    if new_department and not _validate_department(new_department):
        return {"status": "error", "message": "invalid department"}

    if patient_phone_raw and not patient_phone:
        if not (patient_identifier or hospital_reservation_id or reservation_id):
            return {"status": "error", "message": "invalid patient_phone"}
        patient_phone_raw = None
    if not patient_identifier and (patient_phone_raw or patient_phone):
        patient_identifier = _lookup_patient_identifier_by_phone(patient_phone_raw or patient_phone)
    if not new_time_text and not new_department and not (doctor_name or doctor_id):
        return {"status": "error", "message": "new_time or new_department required"}

    hospital_qs = _get_hospital_reservations_qs()
    if hospital_qs is not None and (hospital_reservation_id or patient_identifier):
        try:
            record = None
            if hospital_reservation_id:
                record = hospital_qs.filter(id=hospital_reservation_id).first()
            elif patient_identifier:
                now = timezone.now()
                record = (
                    hospital_qs.filter(
                        patient_identifier=patient_identifier,
                        start_time__gte=now,
                    )
                    .exclude(status__iexact="cancelled")
                    .order_by("start_time")
                    .first()
                )
            if record:
                base_dt = record.start_time
                if timezone.is_naive(base_dt):
                    base_dt = timezone.make_aware(base_dt, timezone.get_current_timezone())
                new_dt = None
                end_time = None
                if new_time_text:
                    time_explicit = _has_time_component(new_time_text)
                    new_dt = _build_rescheduled_datetime(new_time_text, base_dt)
                    if not new_dt:
                        return {"status": "error", "message": "invalid new_time"}
                    now_local = timezone.localtime(timezone.now())
                    if new_dt < now_local:
                        return {
                            "status": "error",
                            "message": "past datetime",
                            "reply_text": "지난 날짜나 시간으로는 예약을 변경할 수 없습니다. 오늘 이후의 날짜와 시간을 알려주세요.",
                        }
                    if not _is_clinic_open_datetime(new_dt):
                        if not time_explicit and not _is_closed_clinic_date(new_dt.date()):
                            return {
                                "status": "error",
                                "message": "time_required",
                                "reply_text": TIME_REQUIRED_REPLY,
                            }
                        return {
                            "status": "error",
                            "message": "closed_hours",
                            "reply_text": CLINIC_CLOSED_REPLY,
                        }
                    if record.end_time:
                        end_time = record.end_time
                        if timezone.is_naive(end_time):
                            end_time = timezone.make_aware(end_time, timezone.get_current_timezone())
                        duration = end_time - base_dt
                        if duration.total_seconds() <= 0:
                            duration = timedelta(minutes=30)
                        end_time = new_dt + duration
                    else:
                        end_time = new_dt + timedelta(minutes=30)
                doctor_info = None
                doctor_department = new_department or record.doctor_department
                if doctor_name or doctor_id:
                    doctor_info = _resolve_doctor_info(
                        doctor_department or new_department or "",
                        doctor_name,
                        doctor_id,
                    )
                    if not doctor_info:
                        return {
                            "status": "not_found",
                            "message": "doctor not found",
                            "reply_text": "요청하신 의료진 정보를 찾지 못했습니다. 의료진 이름을 다시 알려주세요.",
                        }
                elif new_department:
                    doctor_info = _get_default_doctor_info(new_department)
                now = timezone.localtime(timezone.now())
                updates = []
                params: list[Any] = []
                if new_dt is not None:
                    updates.append("start_time = %s")
                    params.append(new_dt)
                    updates.append("end_time = %s")
                    params.append(end_time)
                if new_department:
                    updates.append("doctor_department = %s")
                    params.append(new_department)
                    updates.append("title = %s")
                    params.append(f"{new_department} 진료 예약")
                if doctor_info:
                    updates.append("doctor_code = %s")
                    params.append(doctor_info["doctor_code"])
                    updates.append("doctor_username = %s")
                    params.append(doctor_info["doctor_username"])
                    updates.append("doctor_name = %s")
                    params.append(doctor_info["doctor_name"])
                    updates.append("doctor_id = %s")
                    params.append(doctor_info["doctor_id"])
                updates.append("updated_at = %s")
                params.append(now)
                params.append(record.id)
                with connections["hospital"].cursor() as cursor:
                    cursor.execute(
                        f"UPDATE patients_appointment SET {', '.join(updates)} WHERE id = %s",
                        params,
                    )
                payload = _hospital_reservation_payload(record)
                if new_dt is not None:
                    payload["scheduled_at"] = new_dt.isoformat()
                    payload["requested_time"] = new_dt.isoformat()
                    payload["end_time"] = end_time.isoformat() if end_time else payload.get("end_time")
                if new_department:
                    payload["department"] = new_department
                    payload["title"] = f"{new_department} 진료 예약"
                if doctor_info:
                    payload["doctor_name"] = doctor_info["doctor_name"]
                doctor_display = None
                doctor_reply_name = None
                if payload.get("doctor_name"):
                    doctor_display = _format_doctor_display_name(
                        payload.get("doctor_name"), doctor_info.get("doctor_code") if doctor_info else None
                    )
                    doctor_reply_name = _format_doctor_reply_name(payload.get("doctor_name"))
                if doctor_info:
                    logger.info(
                        "reservation reschedule doctor: department=%s doctor_code=%s doctor_name=%s",
                        payload.get("department") or new_department or "-",
                        doctor_info.get("doctor_code"),
                        doctor_info.get("doctor_name"),
                    )
                date_text, time_text = _format_schedule_text(payload.get("scheduled_at"))
                doctor_suffix = (
                    f" {doctor_reply_name} 의료진으로" if doctor_reply_name else ""
                )
                if new_department and new_dt is not None:
                    reply_text = (
                        f"예약을 {date_text} {time_text} {new_department}{doctor_suffix} 변경했습니다. "
                        "변경 사항이 맞는지 확인해 주세요."
                    )
                elif new_department:
                    reply_text = (
                        f"예약 진료과를 {new_department}{doctor_suffix} 변경했습니다. "
                        f"일정은 {date_text} {time_text}입니다."
                    )
                elif doctor_info:
                    reply_text = (
                        f"예약 의료진을 {doctor_reply_name}으로 변경했습니다. "
                        f"일정은 {date_text} {time_text}입니다."
                    )
                else:
                    reply_text = (
                        f"예약을 {date_text} {time_text}로 변경했습니다. 변경 사항이 맞는지 확인해 주세요."
                    )
                reservation_table = _build_reservation_table_data([payload])
                return {
                    "status": "ok",
                    "reservation": payload,
                    "reply_text": reply_text,
                    "table": reservation_table,
                }
        except Exception as exc:  # pragma: no cover
            logger.warning("hospital reservation reschedule failed: %s", exc)

    qs = Reservation.objects.all()
    if reservation_id:
        qs = qs.filter(id=reservation_id)
    elif patient_phone:
        qs = qs.filter(patient_phone__in={patient_phone_raw or "", patient_phone})
    else:
        return {"status": "error", "message": "reservation_id or patient_phone required"}

    reservation = qs.order_by("-created_at").first()
    if not reservation:
        return {"status": "not_found"}

    update_fields = ["updated_at"]
    if new_time_text:
        if _has_date_component(new_time_text) and not _has_time_component(new_time_text):
            return {
                "status": "error",
                "message": "preferred_time required",
                "reply_text": TIME_REQUIRED_REPLY,
            }
        parsed_dt = _parse_preferred_datetime(new_time_text)
        if not parsed_dt and any(word in new_time_text for word in ASAP_TIME_WORDS):
            parsed_dt = _build_asap_datetime()
        if parsed_dt:
            now = timezone.localtime(timezone.now())
            if parsed_dt < now:
                return {
                    "status": "error",
                    "message": "past datetime",
                    "reply_text": "지난 날짜나 시간으로는 예약을 변경할 수 없습니다. 오늘 이후의 날짜와 시간을 알려주세요.",
                }
        if parsed_dt and not _is_clinic_open_datetime(parsed_dt):
            return {
                "status": "error",
                "message": "closed_hours",
                "reply_text": CLINIC_CLOSED_REPLY,
            }
        reservation.requested_time_text = new_time_text
        update_fields.append("requested_time_text")
    if new_department:
        reservation.department = new_department
        update_fields.append("department")
    reservation.save(update_fields=update_fields)
    payload = _reservation_payload(reservation)
    raw_time = payload.get("requested_time") or ""
    date_text, time_text = _format_schedule_text(raw_time)
    if new_department and new_time_text:
        reply_text = (
            f"예약을 {date_text} {time_text} {new_department}로 변경했습니다. "
            "변경 사항이 맞는지 확인해 주세요."
        )
    elif new_department:
        reply_text = f"예약 진료과를 {new_department}로 변경했습니다. 일정은 {date_text} {time_text}입니다."
    else:
        reply_text = f"예약을 {date_text} {time_text}로 변경했습니다. 변경 사항이 맞는지 확인해 주세요."
    reservation_table = _build_reservation_table_data([payload])
    return {
        "status": "ok",
        "reservation": payload,
        "reply_text": reply_text,
        "table": reservation_table,
    }


def _reservation_history(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    if not _has_auth_context(context):
        return {
            "status": "auth_required",
            "message": "auth required",
            "reply_text": AUTH_REQUIRED_REPLY,
        }
    limit = int(args.get("limit") or 5)
    limit = max(1, min(limit, 20))
    offset = _parse_int(args.get("offset") or args.get("cursor")) or 0
    offset = max(0, offset)
    reply_style = (args.get("reply_style") or args.get("format") or "").strip().lower()
    label = args.get("label") or "예약"
    patient_phone_raw = args.get("patient_phone")
    patient_phone = _normalize_phone(patient_phone_raw)
    patient_identifier = args.get("patient_identifier") or args.get("patient_id")
    if not patient_identifier and context and context.metadata:
        patient_identifier = _get_patient_id(context.metadata)
    if not patient_phone_raw and context and context.metadata:
        meta_phone = _get_metadata_value(context.metadata, ["patient_phone", "phone", "tel"])
        if isinstance(meta_phone, str) and meta_phone.strip():
            patient_phone_raw = meta_phone
            patient_phone = _normalize_phone(meta_phone)
    session_id = args.get("session_id") or (context.session_id if context else None)
    if patient_phone_raw and not patient_phone:
        if not (patient_identifier or session_id):
            return {"status": "error", "message": "invalid patient_phone"}
        patient_phone_raw = None
    if not patient_identifier and patient_phone_raw:
        patient_identifier = _lookup_patient_identifier_by_phone(patient_phone_raw)

    def _build_hospital_history(base_qs: Any) -> Dict[str, Any]:
        base = base_qs.filter(patient_identifier=patient_identifier).exclude(status__iexact="cancelled")
        now = timezone.now()
        if reply_style == "single":
            upcoming = list(base.filter(start_time__gte=now).order_by("start_time")[: offset + 1])
            record = upcoming[offset] if len(upcoming) > offset else None
            if not record:
                if offset > 0:
                    message = "다음 예약이 없습니다. 다른 일정이 필요하시면 알려주세요."
                else:
                    message = (
                        "현재 예정된 예약이 없습니다. 원하시면 예약을 도와드리겠습니다. "
                        "진료과와 희망 날짜/시간을 알려주세요."
                    )
                return {"status": "ok", "reservations": [], "reply_text": message}
            payload = _hospital_reservation_payload(record)
            table = _build_reservation_table_data([payload])
            return {
                "status": "ok",
                "reservations": [payload],
                "reply_text": _format_reservation_single(payload, str(label)),
                "table": table,
            }
        records = list(base.filter(start_time__gte=now).order_by("start_time")[:limit])
        items = [_hospital_reservation_payload(r) for r in records]
        table = _build_reservation_table_data(items)
        return {
            "status": "ok",
            "reservations": items,
            "reply_text": _build_reservation_table(items),
            "table": table,
        }

    hospital_qs = _get_hospital_reservations_qs()
    if hospital_qs is not None and patient_identifier:
        try:
            response = _build_hospital_history(hospital_qs)
            if response.get("reservations"):
                return response
            fallback_qs = _get_hospital_reservations_qs_for_alias("default")
            if fallback_qs is not None and getattr(fallback_qs, "_db", "default") != getattr(
                hospital_qs, "_db", "hospital"
            ):
                fallback_response = _build_hospital_history(fallback_qs)
                if fallback_response.get("reservations"):
                    return fallback_response
            return response
        except Exception as exc:  # pragma: no cover
            logger.warning("hospital reservation history failed: %s", exc)
            fallback_qs = _get_hospital_reservations_qs_for_alias("default")
            if fallback_qs is not None:
                try:
                    fallback_response = _build_hospital_history(fallback_qs)
                    if fallback_response.get("reservations") or reply_style == "single":
                        return fallback_response
                except Exception as fallback_exc:  # pragma: no cover
                    logger.warning("hospital reservation history fallback failed: %s", fallback_exc)

    if patient_identifier and not patient_phone and not session_id:
        if reply_style == "single":
            if offset > 0:
                message = "다음 예약이 없습니다. 다른 일정이 필요하시면 알려주세요."
            else:
                message = (
                    "현재 예정된 예약이 없습니다. 원하시면 예약을 도와드리겠습니다. "
                    "진료과와 희망 날짜/시간을 알려주세요."
                )
            return {"status": "ok", "reservations": [], "reply_text": message}
        return {"status": "ok", "reservations": [], "reply_text": _build_reservation_table([])}

    qs = Reservation.objects.exclude(status="cancelled")
    if patient_phone:
        qs = qs.filter(patient_phone__in={patient_phone_raw or "", patient_phone})
    elif session_id:
        qs = qs.filter(session_id=session_id)
    else:
        return {"status": "error", "message": "patient_id, patient_phone, or session_id required"}

    records = list(qs.order_by("-created_at"))
    now = timezone.now()
    sortable: list[tuple[datetime | None, datetime, Reservation]] = []
    for record in records:
        scheduled_at = _resolve_requested_datetime(record.requested_time_text)
        if scheduled_at and scheduled_at < now:
            continue
        sortable.append((scheduled_at, record.created_at, record))
    sortable.sort(key=lambda item: (item[0] is None, item[0] or item[1]))
    sorted_records = [item[2] for item in sortable]
    if reply_style == "single":
        if len(sorted_records) <= offset:
            if offset > 0:
                message = "다음 예약이 없습니다. 다른 일정이 필요하시면 알려주세요."
            else:
                message = (
                    "현재 예정된 예약이 없습니다. 원하시면 예약을 도와드리겠습니다. "
                    "진료과와 희망 날짜/시간을 알려주세요."
                )
            return {"status": "ok", "reservations": [], "reply_text": message}
        payload = _reservation_payload(sorted_records[offset])
        table = _build_reservation_table_data([payload])
        return {
            "status": "ok",
            "reservations": [payload],
            "reply_text": _format_reservation_single(payload, str(label)),
            "table": table,
        }
    items = [_reservation_payload(r) for r in sorted_records[:limit]]
    table = _build_reservation_table_data(items)
    return {
        "status": "ok",
        "reservations": items,
        "reply_text": _build_reservation_table(items),
        "table": table,
    }


def _medical_history(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    limit = int(args.get("limit") or 5)
    limit = max(1, min(limit, 20))
    patient_identifier = args.get("patient_identifier") or args.get("patient_id")
    patient_phone_raw = args.get("patient_phone")
    if not patient_identifier:
        patient_identifier = _lookup_patient_identifier_by_phone(patient_phone_raw)
    if not patient_identifier:
        return {"status": "error", "message": "patient_id required"}
    try:
        with connections["hospital"].cursor() as cursor:
            cursor.execute(
                "SELECT id, patient_id, name, department, status, notes, reception_start_time, treatment_end_time, is_treatment_completed "
                "FROM medical_record WHERE patient_id = %s ORDER BY reception_start_time DESC LIMIT %s",
                [str(patient_identifier), limit],
            )
            rows = cursor.fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("hospital medical history lookup failed: %s", exc)
        return {"status": "error", "message": "medical_history lookup failed"}
    records = []
    for row in rows:
        records.append(
            {
                "id": row[0],
                "patient_id": row[1],
                "name": row[2],
                "department": row[3],
                "status": row[4],
                "notes": row[5] or "",
                "reception_start_time": row[6].isoformat() if row[6] else "",
                "treatment_end_time": row[7].isoformat() if row[7] else "",
                "is_treatment_completed": bool(row[8]),
            }
        )
    if not records:
        return {
            "status": "not_found",
            "records": [],
            "reply_text": "비로그인 상태에서는 예약이 어렵습니다. 로그인 후 이용해 주세요. 전화 예약은 대표번호 1577-3330으로 문의해 주세요.",
        }
    table = _build_medical_history_table_data(records)
    return {
        "status": "ok",
        "records": records,
        "reply_text": _build_medical_history_table(records),
        "table": table,
    }

def _wait_status(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    department = _normalize_department(args.get("department"))
    patient_identifier = args.get("patient_identifier") or args.get("patient_id")
    patient_phone_raw = args.get("patient_phone")
    if not patient_identifier and patient_phone_raw:
        patient_identifier = _lookup_patient_identifier_by_phone(patient_phone_raw)

    def _count_waiting(dept: str) -> int | None:
        try:
            with connections["hospital"].cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM medical_record "
                    "WHERE department = %s "
                    "AND (is_treatment_completed = 0 OR is_treatment_completed IS NULL)",
                    [dept],
                )
                row = cursor.fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("hospital wait count failed: %s", exc)
            return None

    if patient_identifier:
        try:
            with connections["hospital"].cursor() as cursor:
                cursor.execute(
                    "SELECT department, reception_start_time, status, notes "
                    "FROM medical_record "
                    "WHERE patient_id = %s "
                    "AND (is_treatment_completed = 0 OR is_treatment_completed IS NULL) "
                    "ORDER BY reception_start_time DESC LIMIT 1",
                    [str(patient_identifier)],
                )
                row = cursor.fetchone()
                if row:
                    record_department = row[0] or ""
                    start_time = row[1]
                    status = row[2] or ""
                    notes = row[3] or ""
                    department = _normalize_department(record_department) or record_department
                    if start_time:
                        cursor.execute(
                            "SELECT COUNT(*) FROM medical_record "
                            "WHERE department = %s "
                            "AND (is_treatment_completed = 0 OR is_treatment_completed IS NULL) "
                            "AND reception_start_time < %s",
                            [record_department, start_time],
                        )
                        ahead = cursor.fetchone()
                        ahead_count = int(ahead[0]) if ahead else 0
                    else:
                        ahead_count = 0
                    queue_position = ahead_count + 1
                    wait_record = None
                    if department:
                        wait_record = (
                            WaitStatus.objects.filter(department=department)
                            .order_by("-last_updated")
                            .first()
                        )
                    if wait_record:
                        reply_text = (
                            f"{department} 대기순번은 {queue_position}번입니다. "
                            f"현재 대기중인 사람은 {wait_record.current_waiting}명이며, "
                            f"약 {wait_record.estimated_minutes}분 뒤에 진료가 가능합니다."
                        )
                    else:
                        waiting_count = _count_waiting(record_department) if record_department else None
                        if waiting_count is None:
                            reply_text = (
                                f"{department} 대기순번은 {queue_position}번입니다. "
                                "예상 대기 시간은 접수창구에서 확인해 주세요."
                            )
                        else:
                            reply_text = (
                                f"{department} 대기순번은 {queue_position}번입니다. "
                                f"현재 대기중인 사람은 {waiting_count}명입니다."
                            )
                    return {
                        "status": "ok",
                        "patient_id": patient_identifier,
                        "department": department,
                        "queue_position": queue_position,
                        "status_text": status,
                        "notes": notes,
                        "reply_text": reply_text,
                    }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("hospital patient wait lookup failed: %s", exc)
        if department:
            waiting_count = _count_waiting(department)
            wait_record = (
                WaitStatus.objects.filter(department=department)
                .order_by("-last_updated")
                .first()
            )
            if wait_record and waiting_count is not None:
                reply_text = (
                    f"{department} 대기순번에 없습니다. "
                    f"현재 대기중인 사람은 {wait_record.current_waiting}명입니다."
                )
            elif waiting_count is not None:
                reply_text = (
                    f"{department} 대기순번에 없습니다. "
                    f"현재 대기중인 사람은 {waiting_count}명입니다."
                )
            else:
                reply_text = (
                    f"{department} 대기순번에 없습니다. "
                    "접수창구로 문의해 주세요."
                )
            return {
                "status": "not_found",
                "patient_id": patient_identifier,
                "department": department,
                "reply_text": reply_text,
            }
        if not department:
            return {
                "status": "not_found",
                "patient_id": patient_identifier,
                "reply_text": "현재 접수된 진료가 없습니다. 진료과를 알려주시면 대기 현황을 확인해 드리겠습니다.",
            }

    if not department:
        return {
            "status": "error",
            "message": "department required",
            "reply_text": "대기 현황을 확인할 진료과를 알려주세요.",
        }
    if not _validate_department(department):
        return {"status": "error", "message": "invalid department"}

    record = WaitStatus.objects.filter(department=department).order_by("-last_updated").first()
    if not record:
        waiting_count = _count_waiting(department)
        if waiting_count is not None:
            return {
                "status": "not_found",
                "department": department,
                "current_waiting": waiting_count,
                "reply_text": (
                    f"{department} 대기순번에 없습니다. "
                    f"현재 대기중인 사람은 {waiting_count}명입니다."
                ),
            }
        available = list(WaitStatus.objects.values_list("department", flat=True))
        available = [value for value in available if isinstance(value, str) and value.strip()]
        available_text = ", ".join(sorted(set(available)))
        if available_text:
            reply_text = (
                f"{department} 대기 현황 데이터가 없습니다. "
                f"현재 확인 가능한 진료과는 {available_text}입니다."
            )
        else:
            reply_text = (
                f"{department} 대기 현황 데이터를 확인하지 못했습니다. "
                "접수창구로 문의해 주세요."
            )
        return {
            "status": "not_found",
            "department": department,
            "available_departments": available,
            "reply_text": reply_text,
        }

    return {
        "status": "ok",
        "department": record.department,
        "current_waiting": record.current_waiting,
        "estimated_minutes": record.estimated_minutes,
        "last_updated": record.last_updated.isoformat(),
        "reply_text": (
            f"{record.department} 대기중인 사람은 {record.current_waiting}명이며, "
            f"약 {record.estimated_minutes}분 뒤에 진료가 가능합니다."
        ),
    }


def _notification_send(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    channel = _normalize_channel(args.get("channel"))
    message = args.get("message")
    if not channel or not message:
        return {"status": "error", "message": "channel and message required"}

    notification = Notification.objects.create(
        session_id=(context.session_id if context else "") or "",
        channel=channel,
        target=args.get("target") or "",
        message=message,
        schedule_at=_parse_optional_datetime(args.get("schedule_at")),
        status="pending",
    )
    return {
        "status": "ok",
        "notification": {
            "id": notification.id,
            "channel": notification.channel,
            "target": notification.target,
            "status": notification.status,
        },
    }


def _collect_doctors_from_qs(qs: Any, field_names: set[str]) -> list[dict[str, Any]]:
    doctors: list[dict[str, Any]] = []
    for user in qs[:50]:
        base_name = _build_doctor_name(user)
        username = ""
        if "username" in field_names:
            username = getattr(user, "username", "") or ""
        doctor_id = None
        if "doctor_id" in field_names:
            value = getattr(user, "doctor_id", None)
            if isinstance(value, str):
                value = value.strip()
            if value:
                doctor_id = value
        name = _format_doctor_display_name(base_name, doctor_id)
        title = None
        for field in ["title", "job_title", "position"]:
            if field in field_names:
                value = getattr(user, field, "") or ""
                if isinstance(value, str) and value.strip():
                    title = value.strip()
                    break
        phone = None
        for field in ["phone", "phone_number", "tel"]:
            if field in field_names:
                value = getattr(user, field, "") or ""
                if isinstance(value, str) and value.strip():
                    phone = value.strip()
                    break
        doctors.append(
            {
                "name": name,
                "title": title,
                "phone": phone,
                "doctor_id": doctor_id,
                "username": username,
            }
        )
    return doctors


def _format_doctor_list_reply(department: str, doctors: list[dict[str, Any]]) -> str:
    count = len(doctors)
    return f"{department} 의료진에는 {count}명의 의료진이 있습니다."


def _merge_doctor_lists(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set[str] = set()
    output: list[dict[str, Any]] = []
    for doctor in items:
        doctor_id = doctor.get("doctor_id")
        if isinstance(doctor_id, str):
            doctor_id = doctor_id.strip()
        username = (doctor.get("username") or "").strip().lower()
        key = None
        if doctor_id:
            key = f"id:{doctor_id}"
        elif username:
            key = f"user:{username}"
        if key:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        output.append(doctor)
    return output


def _sort_doctor_list(doctors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(doctor: dict[str, Any]) -> tuple:
        name = doctor.get("name") or ""
        base, suffix = _split_doctor_display(name)
        base_norm = _normalize_doctor_name(base) or base.strip()
        doctor_id = doctor.get("doctor_id")
        if isinstance(doctor_id, str):
            doctor_id = doctor_id.strip()
        if not doctor_id and suffix:
            doctor_id = suffix
        has_id = 0 if doctor_id else 1
        username = (doctor.get("username") or "").strip().lower()
        return (has_id, base_norm, str(doctor_id or ""), username)

    return sorted(doctors, key=_key)


def _resolve_doctor_db_aliases(User: Any) -> list[str]:
    aliases: list[str] = []
    default_alias = User._default_manager.db
    if default_alias:
        aliases.append(default_alias)

    preferred_alias = os.getenv("DOCTOR_DB_ALIAS", "").strip()
    if preferred_alias and preferred_alias in connections.databases:
        if preferred_alias in aliases:
            aliases.remove(preferred_alias)
        aliases.insert(0, preferred_alias)

    if "hospital" in connections.databases and "hospital" not in aliases:
        aliases.append("hospital")

    for alias in connections.databases.keys():
        if alias not in aliases:
            aliases.append(alias)

    return aliases


def _resolve_doctor_info(
    department: str,
    doctor_name: str | None = None,
    doctor_id: str | None = None,
) -> Dict[str, Any] | None:
    normalized_dept = re.sub(r"\s+", "", department)
    dept_hex = normalized_dept.encode("utf-8").hex().upper()
    base_name, display_code = _split_doctor_display(doctor_name)
    if not doctor_id and display_code and display_code != "의료진":
        doctor_id = display_code
    normalized_name = _normalize_doctor_name(base_name or doctor_name)
    requested_id = str(doctor_id).strip() if doctor_id else ""
    if not normalized_name and not requested_id:
        return None

    try:
        User = get_user_model()
        aliases = []
        if "hospital" in connections.databases:
            aliases.append("hospital")
        for alias in _resolve_doctor_db_aliases(User):
            if alias not in aliases:
                aliases.append(alias)
    except Exception:
        aliases = ["hospital", "default"]

    for alias in aliases:
        if alias not in connections.databases:
            continue
        try:
            with connections[alias].cursor() as cursor:
                columns = {
                    col.name
                    for col in connections[alias].introspection.get_table_description(
                        cursor, User._meta.db_table
                    )
                }
                if "department" not in columns:
                    continue

                select_cols = [
                    col
                    for col in [
                        "id",
                        "username",
                        "first_name",
                        "last_name",
                        "doctor_id",
                        "department",
                        "title",
                        "job_title",
                        "position",
                    ]
                    if col in columns
                ]
                if "id" not in select_cols:
                    continue

                where_parts = ["HEX(REPLACE(department, ' ', '')) = %s"]
                params: list[Any] = [dept_hex]

                if "department" in columns:
                    where_parts.append("department IS NOT NULL")
                    where_parts.append("department <> ''")
                    where_parts.append("LOWER(department) <> 'admin'")
                    where_parts.append("department <> '원무과'")
                if "is_superuser" in columns:
                    where_parts.append("(is_superuser = 0 OR is_superuser IS NULL)")
                if "is_active" in columns:
                    where_parts.append("is_active = 1")
                if "username" in columns and ADMIN_EXCLUDE_USERNAMES:
                    placeholders = ", ".join(["%s"] * len(ADMIN_EXCLUDE_USERNAMES))
                    where_parts.append(f"username NOT IN ({placeholders})")
                    params.extend(sorted(ADMIN_EXCLUDE_USERNAMES))

                name_parts: list[str] = []
                if normalized_name:
                    if "last_name" in columns and "first_name" in columns:
                        name_parts.append(
                            "REPLACE(CONCAT(last_name, first_name), ' ', '') = %s"
                        )
                        params.append(normalized_name)
                    if "username" in columns:
                        name_parts.append("username = %s")
                        params.append(doctor_name or "")
                if requested_id:
                    id_parts: list[str] = []
                    if "doctor_id" in columns:
                        id_parts.append("doctor_id = %s")
                        params.append(requested_id)
                    if requested_id.isdigit():
                        id_parts.append("id = %s")
                        params.append(int(requested_id))
                    if id_parts:
                        name_parts.append(f"({' OR '.join(id_parts)})")

                if not name_parts:
                    continue
                where_parts.append(f"({' OR '.join(name_parts)})")

                sql = (
                    f"SELECT {', '.join(select_cols)} FROM {User._meta.db_table} "
                    f"WHERE {' AND '.join(where_parts)} LIMIT 1"
                )
                cursor.execute(sql, params)
                row = cursor.fetchone()
                if not row:
                    continue

            idx = {name: i for i, name in enumerate(select_cols)}
            doctor_pk = row[idx["id"]]
            last_name = row[idx["last_name"]] if "last_name" in idx else ""
            first_name = row[idx["first_name"]] if "first_name" in idx else ""
            username = row[idx["username"]] if "username" in idx else ""
            name = f"{last_name or ''}{first_name or ''}".strip() or (username or "담당의")
            code = None
            if "doctor_id" in idx:
                raw_code = row[idx["doctor_id"]]
                if isinstance(raw_code, str) and raw_code.strip():
                    code = raw_code.strip()
            if not code:
                code = os.getenv("HOSPITAL_DEFAULT_DOCTOR_CODE", "").strip()
            if not code:
                try:
                    code = f"D{timezone.localdate().year}{int(doctor_pk):03d}"
                except Exception:
                    code = ""

            return {
                "doctor_id": int(doctor_pk) if str(doctor_pk).isdigit() else doctor_pk,
                "doctor_code": code,
                "doctor_username": username or str(doctor_pk),
                "doctor_name": name,
                "doctor_department": department,
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("doctor resolve failed: alias=%s err=%s", alias, exc)

    return None


def _build_doctor_table(doctors: list[dict[str, Any]]) -> Dict[str, List[List[str]]]:
    headers = ["이름", "직책", "연락처"]
    rows: List[List[str]] = []
    for doctor in doctors:
        name = doctor.get("name") or "-"
        title = doctor.get("title") or "-"
        phone = doctor.get("phone") or "-"
        rows.append([name, title, phone])
    return {"headers": headers, "rows": rows}


def _doctor_list_via_sql(User: Any, department: str, db_alias: str | None = None) -> list[dict[str, Any]]:
    alias = db_alias or User._default_manager.db
    table = User._meta.db_table
    normalized_dept = re.sub(r"\s+", "", department)
    dept_hex = normalized_dept.encode("utf-8").hex().upper()
    try:
        with connections[alias].cursor() as cursor:
            columns = {
                col.name
                for col in connections[alias].introspection.get_table_description(cursor, table)
            }
            if "department" not in columns:
                return []

            select_cols = [
                col
                for col in [
                    "username",
                    "first_name",
                    "last_name",
                    "doctor_id",
                    "title",
                    "job_title",
                    "position",
                    "phone",
                    "phone_number",
                    "tel",
                ]
                if col in columns
            ]
            if not select_cols:
                return []

            where_parts = ["HEX(REPLACE(department, ' ', '')) = %s"]
            params: list[Any] = [dept_hex]

            if "department" in columns:
                where_parts.append("department IS NOT NULL")
                where_parts.append("department <> ''")
                where_parts.append("LOWER(department) <> 'admin'")
                where_parts.append("department <> '원무과'")
            if "is_superuser" in columns:
                where_parts.append("(is_superuser = 0 OR is_superuser IS NULL)")
            if "is_active" in columns:
                where_parts.append("is_active = 1")
            if "username" in columns and ADMIN_EXCLUDE_USERNAMES:
                placeholders = ", ".join(["%s"] * len(ADMIN_EXCLUDE_USERNAMES))
                where_parts.append(f"username NOT IN ({placeholders})")
                params.extend(sorted(ADMIN_EXCLUDE_USERNAMES))

            where_clause = " AND ".join(where_parts)
            order_fields = []
            for field in ["last_name", "first_name", "doctor_id", "username"]:
                if field in select_cols:
                    order_fields.append(field)
            order_clause = f" ORDER BY {', '.join(order_fields)}" if order_fields else ""
            sql = (
                f"SELECT {', '.join(select_cols)} FROM {table} "
                f"WHERE {where_clause}{order_clause} LIMIT 50"
            )
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        idx = {name: i for i, name in enumerate(select_cols)}
        doctors: list[dict[str, Any]] = []
        for row in rows:
            last_name = row[idx["last_name"]] if "last_name" in idx else ""
            first_name = row[idx["first_name"]] if "first_name" in idx else ""
            username = row[idx["username"]] if "username" in idx else ""
            name = f"{last_name or ''}{first_name or ''}".strip() or (username or "담당의")
            doctor_id = row[idx["doctor_id"]] if "doctor_id" in idx else None
            if isinstance(doctor_id, str):
                doctor_id = doctor_id.strip() or None
            title = None
            for key in ["title", "job_title", "position"]:
                if key in idx:
                    value = row[idx[key]]
                    if isinstance(value, str) and value.strip():
                        title = value.strip()
                        break
            phone = None
            for key in ["phone", "phone_number", "tel"]:
                if key in idx:
                    value = row[idx[key]]
                    if isinstance(value, str) and value.strip():
                        phone = value.strip()
                        break
            name = _format_doctor_display_name(name, doctor_id)
            doctors.append(
                {
                    "name": name,
                    "title": title,
                    "phone": phone,
                    "doctor_id": doctor_id,
                    "username": username,
                }
            )
        return doctors
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("doctor list sql fallback failed: alias=%s err=%s", alias, exc)
        return []


def _doctor_list(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    department = _normalize_department(args.get("department"))
    if not department and context and context.metadata:
        meta_dept = _get_metadata_value(
            context.metadata,
            ["department", "dept", "진료과", "last_department", "recent_department"],
        )
        department = _normalize_department(meta_dept)
    if not department and context and context.session_id:
        recent_messages = list(
            ChatMessage.objects.filter(session_id=context.session_id).order_by("-created_at")[:5]
        )
        for message in recent_messages:
            for text in [message.user_question, message.bot_answer]:
                if not text:
                    continue
                inferred = _extract_department(text, None)
                if inferred:
                    department = inferred
                    break
            if department:
                break
    if not department:
        return {
            "status": "error",
            "message": "department required",
            "reply_text": "어느 진료과 의료진을 찾으시나요?",
        }
    if department.lower() in EXCLUDED_DOCTOR_DEPARTMENTS:
        return {
            "status": "not_found",
            "department": department,
            "doctors": [],
            "reply_text": "해당 진료과 의료진 정보를 찾지 못했습니다. 원하시면 진료과명을 정확히 알려주세요.",
        }

    try:
        User = get_user_model()
        field_names = {field.name for field in User._meta.get_fields()}
        if "department" not in field_names:
            all_doctors: list[dict[str, Any]] = []
            for alias in _resolve_doctor_db_aliases(User):
                alias_doctors = _doctor_list_via_sql(User, department, alias)
                if alias_doctors:
                    logger.info(
                        "doctor list sql: alias=%s department=%s count=%s",
                        alias,
                        department,
                        len(alias_doctors),
                    )
                    all_doctors.extend(alias_doctors)
            doctors = _merge_doctor_lists(all_doctors)
            doctors = _sort_doctor_list(doctors)
            if not doctors:
                return {
                    "status": "not_found",
                    "department": department,
                    "doctors": [],
                    "reply_text": "해당 진료과 의료진 정보를 찾지 못했습니다. 원하시면 진료과명을 정확히 알려주세요.",
                }
            reply_text = _format_doctor_list_reply(department, doctors)
            table = _build_doctor_table(doctors)
            logger.info(
                "doctor list result: department=%s count=%s sample=%s",
                department,
                len(doctors),
                ", ".join([doctor.get("name") or "-" for doctor in doctors[:3]]),
            )
            return {
                "status": "ok",
                "department": department,
                "doctors": doctors,
                "reply_text": reply_text,
                "table": table,
            }

        normalized = re.sub(r"\s+", "", department)
        qs = User.objects.annotate(
            dept_compact=Replace("department", Value(" "), Value(""))
        ).filter(Q(department__iexact=department) | Q(dept_compact__iexact=normalized))
        qs = qs.exclude(department__isnull=True).exclude(department__exact="")
        qs = qs.exclude(department__iexact="admin").exclude(department__iexact="원무과")
        if "is_superuser" in field_names:
            qs = qs.filter(is_superuser=False)
        if "is_active" in field_names:
            qs = qs.filter(is_active=True)
        if "username" in field_names:
            qs = qs.exclude(username__in=ADMIN_EXCLUDE_USERNAMES)

        order_fields = [field for field in ["last_name", "first_name", "username"] if field in field_names]
        if order_fields:
            qs = qs.order_by(*order_fields)

        doctors = _collect_doctors_from_qs(qs, field_names)
        doctors = _sort_doctor_list(doctors)
        if not doctors:
            doctors = _collect_doctors_from_qs(qs, field_names)
            doctors = _sort_doctor_list(doctors)

        if not doctors:
            return {
                "status": "not_found",
                "department": department,
                "doctors": [],
                "reply_text": "해당 진료과 의료진 정보를 찾지 못했습니다. 원하시면 진료과명을 정확히 알려주세요.",
            }

        reply_text = _format_doctor_list_reply(department, doctors)
        
        # 검색 결과 모호성 처리: 결과가 너무 많을 경우(6명 이상) 이름 검색 유도
        if len(doctors) >= 6:
            reply_text += (
                f"\n\n검색된 의료진이 {len(doctors)}명입니다. "
                "찾으시는 선생님 성함을 말씀해 주시면 더 빠르게 안내해 드릴 수 있습니다."
            )

        table = _build_doctor_table(doctors)
        logger.info(
            "doctor list result: department=%s count=%s sample=%s",
            department,
            len(doctors),
            ", ".join([doctor.get("name") or "-" for doctor in doctors[:3]]),
        )
        return {
            "status": "ok",
            "department": department,
            "doctors": doctors,
            "reply_text": reply_text,
            "table": table,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("doctor list lookup failed: %s", exc)
        return {
            "status": "error",
            "message": "doctor lookup failed",
            "reply_text": "의료진 정보를 확인하는 데 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        }


def _session_history(args: Dict[str, Any], context: ToolContext | None) -> Dict[str, Any]:
    limit = int(args.get("limit") or 5)
    limit = max(1, min(limit, 20))
    session_id = args.get("session_id") or (context.session_id if context else None)
    if not session_id:
        return {"status": "error", "message": "session_id required"}

    items = []
    for item in ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:limit]:
        items.append(
            {
                "question": item.user_question,
                "answer": item.bot_answer,
                "created_at": item.created_at.isoformat(),
            }
        )
    return {"status": "ok", "session_id": session_id, "messages": items}


def _reservation_payload(reservation: Reservation) -> Dict[str, Any]:
    doctor_name = getattr(reservation, "doctor_name", "") or ""
    return {
        "id": reservation.id,
        "session_id": reservation.session_id,
        "patient_name": reservation.patient_name,
        "patient_phone": reservation.patient_phone,
        "department": reservation.department,
        "doctor_name": doctor_name,
        "requested_time": reservation.requested_time_text,
        "memo": (reservation.reason or "").strip(),
        "status": reservation.status,
        "channel": reservation.channel,
        "created_at": reservation.created_at.isoformat(),
    }


def _parse_optional_datetime(value: Any) -> Any:
    if not value:
        return None
    if isinstance(value, str):
        try:
            return timezone.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def format_tool_result(result: Dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
