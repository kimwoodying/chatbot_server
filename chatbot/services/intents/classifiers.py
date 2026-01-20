# chatbot/services/intents/classifiers.py
import re
from typing import Dict, Any, List

from chatbot.models import ChatMessage
from chatbot.services.tooling import _extract_department
from chatbot.services.intents.keywords import (
    SMALLTALK_EXCLUDE_KEYWORDS,
    DOCTOR_FOLLOWUP_CUES,
    SYMPTOM_DEPARTMENT_RULES,
    TIME_HINT_WORDS,
    TIME_HINT_PATTERN,
    SYMPTOM_BOOKING_CUES,
    SYMPTOM_VISIT_CUES,
    RESCHEDULE_CUES,
    DOCTOR_CHANGE_CUES,
    DOCTOR_CHANGE_PROMPT,
    DOCTOR_SELECT_PROMPT,
    DOCTOR_SELECT_CUES,
    NEGATIVE_CUES,
    CANCEL_CUES,
    BULK_CANCEL_CUES,
    BOOKING_PROMPT_MARKERS,
    TIME_EXTRACT_PATTERN,
    SYMPTOM_INFO_CUES,
    SYMPTOM_INTENT_CUES,
    RESERVATION_LOGIN_GUARD_CUES,
)


def is_smalltalk_query(query: str) -> bool:
    if not query:
        return False
    if any(keyword in query for keyword in SMALLTALK_EXCLUDE_KEYWORDS):
        return False
    if re.search(r"\d", query):
        return False
    compact = re.sub(r"[\s\W_]+", "", query).lower()
    if not compact:
        return False
    if re.fullmatch(r"[ㅎㅋ]+", compact):
        return True
    if compact in {"안녕", "안녕하세요", "하이", "hello", "hi", "반가워", "좋은아침", "좋은오후", "좋은저녁"}:
        return True
    if compact in {"ㅎㅎ", "ㅋㅋ", "ㅎ", "ㅋ", "ㅇㅇ", "ㅇㅋ"}:
        return True
    return False

def is_fixed_info_query(query: str) -> bool:
    if not query:
        return False
    q = query
    q_lower = q.lower()
    is_fixed = any(
        token in q
        for token in [
            "대표번호",
            "전화번호",
            "연락처",
            "응급실",
            "위치",
            "주소",
            "주차",
            "정산",
            "정산소",
            "진료시간",
            "진료 시간",
            "운영시간",
            "운영 시간",
            "접수시간",
            "접수 시간",
            "콜센터",
            "암센터",
        ]
    ) or "parking" in q_lower

    # '내 진료시간', '언제 진료', '나의 예약' 등은 고정 정보가 아니라 개인 예약 조회임
    if is_fixed and any(x in q for x in ["내", "나의", "언제", "예약", "몇 시", "몇시"]):
         # 단, '예약 방법', '예약 안내' 같은 건 고정 정보일 수 있으나 여기서는 Tool 흐름도 괜찮음
         return False
    
    return is_fixed

def is_doctor_followup(query: str, session_id: str | None) -> bool:
    if not query or not session_id:
        return False
    if not any(cue in query for cue in DOCTOR_FOLLOWUP_CUES):
        return False
    last_message = (
        ChatMessage.objects.filter(session_id=session_id)
        .order_by("-created_at")
        .first()
    )
    if not last_message:
        return False
    last_answer = last_message.bot_answer or ""
    return any(keyword in last_answer for keyword in ["의료진", "의사", "교수", "선생님"])

def is_doctor_department_followup(query: str, session_id: str | None) -> bool:
    if not query or not session_id:
        return False
    if not _extract_department(query, None):
        return False
    last_message = (
        ChatMessage.objects.filter(session_id=session_id)
        .order_by("-created_at")
        .first()
    )
    if not last_message:
        return False
    last_answer = last_message.bot_answer or ""
    return "의료진" in last_answer and "진료과" in last_answer

def match_symptom_department(query: str) -> str | None:
    q = (query or "").lower()
    for keywords, department in SYMPTOM_DEPARTMENT_RULES:
        if any(k in q for k in keywords):
            return department
    return None

def has_time_hint(query: str) -> bool:
    if any(word in query for word in TIME_HINT_WORDS):
        return True
    if TIME_HINT_PATTERN.search(query):
        return True
    return any(token in query.lower() for token in ["am", "pm"])

def has_booking_intent(query: str) -> bool:
    return any(k in query for k in SYMPTOM_BOOKING_CUES)


def has_symptom_time_booking_intent(query: str) -> bool:
    if not query or not has_time_hint(query):
        return False
    from chatbot.services.flows import match_symptom_guide
    if match_symptom_department(query) or match_symptom_guide(query):
        return True
    return any(cue in query for cue in SYMPTOM_VISIT_CUES)


def has_additional_booking_intent(query: str) -> bool:
    if not query:
        return False
    if "예약도" in query or "예약도해" in query or "예약도 해" in query:
        return True
    return any(
        phrase in query
        for phrase in [
            "추가 예약",
            "예약 추가해",
            "예약 추가해줘",
            "예약 추가해 줘",
            "추가해줘",
            "추가해 줘",
            "추가해",
            "하나 더",
            "한번 더",
            "또 예약",
            "더 예약",
            "추가로 예약",
        ]
    )


def has_reschedule_cue(query: str) -> bool:
    return any(k in query for k in RESCHEDULE_CUES)


def has_doctor_change_cue(query: str) -> bool:
    if not query:
        return False
    return any(k in query for k in DOCTOR_CHANGE_CUES)


def is_doctor_change_prompt(text: str) -> bool:
    if not text:
        return False
    return DOCTOR_CHANGE_PROMPT in text


def is_doctor_select_prompt(text: str) -> bool:
    if not text:
        return False
    if is_doctor_change_prompt(text):
        return False
    return DOCTOR_SELECT_PROMPT in text


def is_negative_reply(query: str) -> bool:
    if not query:
        return False
    compact = re.sub(r"\s+", "", query.lower())
    return any(cue in compact for cue in NEGATIVE_CUES)


def has_cancel_cue(query: str) -> bool:
    return any(k in query for k in CANCEL_CUES)


def has_bulk_cancel_cue(query: str) -> bool:
    if not query or not has_cancel_cue(query):
        return False
    if any(word in query for word in BULK_CANCEL_CUES):
        return True
    compact = query.replace(" ", "")
    if any(phrase in compact for phrase in ["다취소", "전부취소", "모두취소", "전체취소", "일괄취소"]):
        return True
    if " 다" in query:
        return True
    return False


def is_reservation_summary(text: str) -> bool:
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


def has_department_confirmation_cue(query: str) -> bool:
    return False


def is_booking_prompt(text: str) -> bool:
    if not text:
        return False
    return any(marker in text for marker in BOOKING_PROMPT_MARKERS)


def needs_reservation_login_guard(query: str) -> bool:
    if not query:
        return False
    return any(cue in query for cue in RESERVATION_LOGIN_GUARD_CUES)


def is_wait_department_prompt(text: str) -> bool:
    if not text:
        return False
    markers = [
        "대기 현황을 확인할 진료과",
        "진료과를 알려주시면 대기 현황",
        "대기 현황을 확인해 드리겠습니다",
    ]
    return any(marker in text for marker in markers)


def is_symptom_department_request(query: str) -> bool:
    if any(k in query for k in SYMPTOM_INFO_CUES):
        return False
    # _extract_department is imported from tooling
    if (has_booking_intent(query) or has_symptom_time_booking_intent(query)) and _extract_department(query, None):
        return False
    # 증상 키워드가 있으면 증상 질문으로 간주
    if any(k in query for k in ["멍울", "통증", "아파", "증상", "아프", "불편", "괴로워", "힘들", "출혈"]):
        return True
    if any(k in query for k in SYMPTOM_INTENT_CUES):
        return True
    if any(k in query for k in ["예약", "예약해", "예약 해", "예약잡", "예약 잡", "예약하고", "예약 하고"]):
        from chatbot.services.flows import match_symptom_guide
        return (
            match_symptom_department(query) is not None
            or match_symptom_guide(query) is not None
        )
    return False
