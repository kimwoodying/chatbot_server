from __future__ import annotations

import re
from typing import Any, Dict, Optional

from django.utils import timezone

from chatbot.models import ChatMessage
from chatbot.services.common import AUTH_REQUIRED_REPLY
from chatbot.services.tooling import (
    ToolContext,
    execute_tool,
    _extract_department,
    _extract_doctor_name,
    _has_auth_context,
    _format_doctor_reply_name,
)
from chatbot.services.intents.keywords import (
    RESERVATION_HISTORY_CUES,
    RESCHEDULE_TIME_KEEP_CUES,
    DOCTOR_CHANGE_PROMPT,
    DOCTOR_SELECT_PROMPT,
    TIME_EXTRACT_PATTERN,
    DAY_ONLY_PATTERN,
    TIME_HINT_WORDS,
)
from chatbot.services.intents.classifiers import (
    has_booking_intent,
    has_reschedule_cue,
    has_doctor_change_cue,
    has_cancel_cue,
    is_wait_department_prompt,
    has_bulk_cancel_cue,
    is_doctor_change_prompt,
    is_doctor_select_prompt,
    is_reservation_summary,
    is_booking_prompt,
    is_negative_reply,
    match_symptom_department,
)
from chatbot.services.flows import match_symptom_guide
from chatbot.services.extraction import (
    maybe_reject_closed_date,
    extract_day_only,
    extract_numeric_day,
    extract_day_only_list,
    extract_time_phrase,
    extract_numeric_hour,
    extract_date_phrase,
    is_multi_date_prompt,
    parse_date_only,
    build_date_same_month,
    build_date_from_base_day,
    merge_date_with_time,
    normalize_preferred_time,
    extract_selected_doctor_name,
    infer_recent_doctor_name,
    infer_recent_department,
    should_reschedule_from_summary,
    build_time_followup_message,
    has_specific_time,
    contains_asap,
)


def _build_tool_context(session_id: str | None, metadata: Dict[str, Any] | None) -> ToolContext:
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


def _has_time_or_date_hint(text: str) -> bool:
    if not text:
        return False
    if TIME_EXTRACT_PATTERN.search(text):
        return True
    if DAY_ONLY_PATTERN.search(text):
        return True
    return any(word in text for word in TIME_HINT_WORDS)


def is_negative_only_reply(query: str, metadata: Dict[str, Any] | None) -> bool:
    if not is_negative_reply(query):
        return False
    if _has_time_or_date_hint(query):
        return False
    if _extract_department(query, metadata):
        return False
    if match_symptom_department(query) or match_symptom_guide(query):
        return False
    if _extract_doctor_name(query, metadata):
        return False
    if has_reschedule_cue(query) or has_cancel_cue(query) or has_doctor_change_cue(query):
        return False
    return True


def handle_reservation_followup(
    query: str,
    session_id: str | None,
    metadata: Dict[str, Any] | None,
) -> dict | None:
    if not session_id:
        return None
    department: str | None = None
    preferred_time: str | None = None
    date_hint: str | None = None
    asap = False
    if not _has_auth_context(metadata):
        if (
            any(cue in query for cue in RESERVATION_HISTORY_CUES)
            or has_cancel_cue(query)
            or has_reschedule_cue(query)
            or has_doctor_change_cue(query)
        ):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
    if has_booking_intent(query):
        closed_reply = maybe_reject_closed_date(query)
        if closed_reply:
            return {"reply": closed_reply, "sources": []}
    if any(cue in query for cue in RESERVATION_HISTORY_CUES):
        if not _has_auth_context(metadata):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        has_time = _has_time_or_date_hint(query) or any(
            marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
        )
        has_explicit_department = _extract_department(query, None) is not None
        if not (
            has_time
            or has_reschedule_cue(query)
            or has_doctor_change_cue(query)
            or has_explicit_department
        ):
            tool_context = _build_tool_context(session_id, metadata)
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
    
    last_message = (
        ChatMessage.objects.filter(session_id=session_id)
        .order_by("-created_at")
        .first()
    )
    if not last_message:
        return None
    last_bot_answer = last_message.bot_answer or ""
    has_reservation_signal = (
        has_booking_intent(query)
        or has_reschedule_cue(query)
        or has_cancel_cue(query)
        or has_doctor_change_cue(query)
        or any(cue in query for cue in RESERVATION_HISTORY_CUES)
        or is_booking_prompt(last_bot_answer)
        or is_wait_department_prompt(last_bot_answer)
        or is_doctor_change_prompt(last_bot_answer)
        or is_doctor_select_prompt(last_bot_answer)
        or is_reservation_summary(last_bot_answer)
        or is_multi_date_prompt(last_bot_answer)
    )
    if not has_reservation_signal:
        return None
    if is_wait_department_prompt(last_bot_answer):
        department = _extract_department(query, metadata)
        if department:
            tool_context = _build_tool_context(session_id, metadata)
            result = execute_tool("wait_status", {"department": department}, tool_context)
            if isinstance(result, dict) and result.get("reply_text"):
                return {"reply": result["reply_text"], "sources": []}
        return {"reply": "대기 현황을 확인할 진료과를 알려주세요.", "sources": []}
    if (
        is_negative_only_reply(query, metadata)
        and any(
            marker in last_bot_answer
            for marker in ["지난 날짜나 시간", "오늘 이후의 날짜와 시간"]
        )
    ):
        return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
    if is_multi_date_prompt(last_bot_answer):
        if is_negative_only_reply(query, metadata):
            return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
        day_only = extract_day_only(query) or extract_numeric_day(query)
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
            day_candidates.extend(extract_day_only_list(text))
        if day_only and day_only not in day_candidates:
            day_candidates.append(day_only)
        day_candidates = sorted(set(day_candidates))
        date_hint = None
        for text in [m.user_question for m in recent_messages] + [m.bot_answer for m in recent_messages]:
            date_hint = extract_date_phrase(text)
            if date_hint:
                break
        base_date = parse_date_only(date_hint) if date_hint else timezone.localdate()
        adjusted = (
            build_date_same_month(base_date, day_only)
            or build_date_from_base_day(base_date, day_only)
        ) if base_date else None
        if adjusted:
            date_hint = f"{adjusted.month}월 {adjusted.day}일"
        preferred_time = extract_time_phrase(query)
        if not preferred_time:
            numeric_hour = extract_numeric_hour(query)
            if numeric_hour is not None:
                preferred_time = f"{numeric_hour}시"
        if not preferred_time:
            for text in [m.user_question for m in recent_messages]:
                preferred_time = extract_time_phrase(text)
                if not preferred_time:
                    numeric_hour = extract_numeric_hour(text)
                    if numeric_hour is not None:
                        preferred_time = f"{numeric_hour}시"
                if preferred_time:
                    break
        preferred_time = merge_date_with_time(preferred_time, date_hint)
        preferred_time = normalize_preferred_time(preferred_time, False)
        department = _extract_department(query, metadata) or infer_recent_department(session_id)
        if not department:
            return {"reply": "예약을 위해 진료과를 알려주세요.", "sources": []}
        if not _has_auth_context(metadata):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        doctor_name = extract_selected_doctor_name(query, metadata) or _extract_doctor_name(query, metadata)
        if not doctor_name:
            for message in recent_messages:
                for text in [message.user_question, message.bot_answer]:
                    doctor_name = _extract_doctor_name(text, metadata)
                    if doctor_name:
                        break
                if doctor_name:
                    break
        if not doctor_name:
            doctor_name = infer_recent_doctor_name(session_id)
        if not doctor_name:
            tool_context = _build_tool_context(session_id, metadata)
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
        tool_context = _build_tool_context(session_id, metadata)
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
        if is_doctor_change_prompt(last_bot_answer) or is_reservation_summary(last_bot_answer):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
    if is_doctor_change_prompt(last_bot_answer):
        if is_negative_only_reply(query, metadata):
            return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
        tool_context = _build_tool_context(session_id, metadata)
        doctor_name = extract_selected_doctor_name(query, metadata)
        if not doctor_name:
            department = _extract_department(query, metadata) or infer_recent_department(session_id)
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
    if is_doctor_select_prompt(last_bot_answer):
        if not _has_auth_context(metadata):
            return {"reply": AUTH_REQUIRED_REPLY, "sources": []}
        if has_cancel_cue(query):
            tool_context = _build_tool_context(session_id, metadata)
            cancel_args = {"cancel_all": True} if has_bulk_cancel_cue(query) else {}
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
        if is_negative_only_reply(query, metadata):
            return {"reply": "알겠습니다. 필요하시면 다시 말씀해 주세요.", "sources": []}
        tool_context = _build_tool_context(session_id, metadata)
        doctor_name = extract_selected_doctor_name(query, metadata)
        if not doctor_name:
            department = (
                _extract_department(query, metadata)
                or _extract_department(last_bot_answer, None)
                or infer_recent_department(session_id)
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
            or infer_recent_department(session_id)
        )
        preferred_time = extract_time_phrase(query)
        if not preferred_time:
            numeric_hour = extract_numeric_hour(query)
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
                    numeric_hour = extract_numeric_hour(text)
                    if numeric_hour is not None:
                        preferred_time = f"{numeric_hour}시"
                asap = asap or contains_asap(text)
                if preferred_time:
                    break
        date_hint = extract_date_phrase(last_bot_answer)
        if not date_hint:
            search_texts = [last_message.user_question] + [m.user_question for m in recent_messages]
            search_texts.extend(m.bot_answer for m in recent_messages)
            for text in search_texts:
                date_hint = extract_date_phrase(text)
                if date_hint:
                    break
        day_only = extract_day_only(query)
        if not day_only and is_multi_date_prompt(last_bot_answer):
            day_only = extract_numeric_day(query)
        day_candidates = []
        for text in [last_message.user_question] + [m.user_question for m in recent_messages]:
            day_candidates.extend(extract_day_only_list(text))
        day_candidates = sorted(set(day_candidates))
        if not date_hint and len(day_candidates) > 1 and not day_only:
            return {
                "reply": "여러 날짜가 있습니다. 예약할 날짜를 하나만 알려주세요.",
                "sources": [],
            }
        if not date_hint and not day_only and len(day_candidates) == 1:
            day_only = day_candidates[0]
        if day_only and not extract_date_phrase(query):
            base_date = parse_date_only(date_hint) if date_hint else timezone.localdate()
            adjusted = (
                build_date_same_month(base_date, day_only)
                or build_date_from_base_day(base_date, day_only)
            ) if base_date else None
            if adjusted:
                date_hint = f"{adjusted.month}월 {adjusted.day}일"
                if preferred_time and extract_date_phrase(preferred_time):
                    preferred_time = date_hint
                else:
                    preferred_time = merge_date_with_time(preferred_time, date_hint)
                    if not preferred_time:
                        preferred_time = date_hint
        preferred_time = merge_date_with_time(preferred_time, date_hint)
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
    if is_reservation_summary(last_bot_answer):
        tool_context = _build_tool_context(session_id, metadata)
        auto_reschedule = should_reschedule_from_summary(query, metadata)
        if has_cancel_cue(query):
            cancel_args = {"cancel_all": True} if has_bulk_cancel_cue(query) else {}
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
        if has_reschedule_cue(query) or auto_reschedule:
            new_department = _extract_department(query, metadata)
            has_time = _has_time_or_date_hint(query) or any(
                marker in query for marker in RESCHEDULE_TIME_KEEP_CUES
            )
            if not has_time and not new_department:
                return {
                    "reply": "예약 변경을 위해 변경할 날짜/시간이나 진료과를 알려주세요.",
                    "sources": [],
                }
            reschedule_args: Dict[str, Any] = {}
            if has_time:
                new_time_text = query
                if _has_time_or_date_hint(query):
                    time_phrase = extract_time_phrase(query)
                    if time_phrase:
                        new_time_text = time_phrase
                recent_messages = list(
                    ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:5]
                )
                date_hint = extract_date_phrase(last_bot_answer)
                if not date_hint:
                    search_texts = [last_message.user_question] + [m.user_question for m in recent_messages]
                    search_texts.extend(m.bot_answer for m in recent_messages)
                    for text in search_texts:
                        date_hint = extract_date_phrase(text)
                        if date_hint:
                            break
                day_only = extract_day_only(query)
                if not day_only and is_multi_date_prompt(last_bot_answer):
                    day_only = extract_numeric_day(query)
                day_candidates = []
                for text in [last_message.user_question] + [m.user_question for m in recent_messages]:
                    day_candidates.extend(extract_day_only_list(text))
                day_candidates = sorted(set(day_candidates))
                if not date_hint and len(day_candidates) > 1 and not day_only:
                    return {
                        "reply": "여러 날짜가 있습니다. 예약할 날짜를 하나만 알려주세요.",
                        "sources": [],
                    }
                if not date_hint and not day_only and len(day_candidates) == 1:
                    day_only = day_candidates[0]
                if day_only and not extract_date_phrase(query):
                    base_date = parse_date_only(date_hint) if date_hint else timezone.localdate()
                    adjusted = (
                        build_date_same_month(base_date, day_only)
                        or build_date_from_base_day(base_date, day_only)
                    ) if base_date else None
                    if adjusted:
                        date_hint = f"{adjusted.month}월 {adjusted.day}일"
                        if extract_date_phrase(new_time_text or ""):
                            new_time_text = date_hint
                        else:
                            new_time_text = merge_date_with_time(new_time_text, date_hint)
                            if not new_time_text:
                                new_time_text = date_hint
                new_time_text = merge_date_with_time(new_time_text, date_hint)
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
    
    # Fallback to general reservation logic if no specific prompt context matched
    if department is None:
        department = _extract_department(query, metadata) or infer_recent_department(session_id)
    if preferred_time is None:
        preferred_time = extract_time_phrase(query)
        if not preferred_time:
            numeric_hour = extract_numeric_hour(query)
            if numeric_hour is not None:
                preferred_time = f"{numeric_hour}시"
    if not date_hint:
        date_hint = extract_date_phrase(query) or extract_date_phrase(last_bot_answer)
    day_only = extract_day_only(query)
    if day_only and not extract_date_phrase(query):
        base_date = parse_date_only(date_hint) if date_hint else timezone.localdate()
        adjusted = (
            build_date_same_month(base_date, day_only)
            or build_date_from_base_day(base_date, day_only)
        ) if base_date else None
        if adjusted:
            date_hint = f"{adjusted.month}월 {adjusted.day}일"
            if preferred_time and extract_date_phrase(preferred_time):
                preferred_time = date_hint
            else:
                preferred_time = merge_date_with_time(preferred_time, date_hint)
                if not preferred_time:
                    preferred_time = date_hint
    preferred_time = merge_date_with_time(preferred_time, date_hint)
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
        tool_context = _build_tool_context(session_id, metadata)
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

    tool_context = _build_tool_context(session_id, metadata)
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
