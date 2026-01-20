# chatbot/services/gemini_client.py
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx

from chatbot.config import get_settings
from chatbot.services.cache_service import (
    CACHE_SCOPE_RAG_CONTEXT,
    get_cached_response,
    hash_text,
    save_cache_response,
)
from chatbot.services.security import sanitize_metadata_for_prompt
from chatbot.services.tooling import ToolContext, execute_tool, format_tool_result, get_tool_definitions

logger = logging.getLogger(__name__)


# Marker type to bypass post-processing for strict tool replies.
from chatbot.services.common import ToolReply, clean_response, format_context








# ---------- Policy: emergency_notice.txt 항상 포함 ----------
def _load_emergency_policy() -> str:
    """
    emergency_notice.txt를 읽어서 system prompt 상단에 붙인다.
    - RAG에서 빼더라도(ingest 제외) 여기서 항상 적용됨
    """
    try:
        # 현재 파일: chatbot/services/gemini_client.py
        # raw 폴더: chatbot/data/raw/emergency_notice.txt (네 프로젝트 기준)
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
        p = raw_dir / "emergency_notice.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("emergency_notice.txt 로드 실패: %s", e)
    return ""


# ---------- LLM 저수준 호출: Gemini ----------
def _call_gemini(system_prompt: str, user_message: str, temperature: float) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY가 설정되어 있지 않습니다.")
        return ""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    headers = {"Content-Type": "application/json"}
    params = {"key": settings.gemini_api_key}
    body = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": user_message},
                ]
            }
        ],
        "generationConfig": {"temperature": temperature},
    }

    max_attempts = 3
    resp = None
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(url, params=params, headers=headers, json=body)
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("Gemini API error %s: %s", status, exc.response.text)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return ""
        except httpx.RequestError as exc:
            logger.error("Gemini request error: %s", exc)
            if attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return ""

    if resp is None:
        return ""

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        return ""
    text = parts[0].get("text")
    return text.strip() if isinstance(text, str) else ""


# ---------- LLM 저수준 호출: OpenAI ----------
def _openai_chat_completion(
    messages: list[dict],
    temperature: float,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    model_override: str | None = None,
) -> dict | None:
    settings = get_settings()
    if not getattr(settings, "openai_api_key", None):
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.openai_api_key}",
    }
    body = {
        "model": model_override or settings.openai_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

    max_attempts = 3
    resp = None
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("OpenAI API error %s: %s", status, exc.response.text)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return None
        except httpx.RequestError as exc:
            logger.error("OpenAI request error: %s", exc)
            if attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return None

    if resp is None:
        return None

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    return choices[0].get("message")


def _call_openai(
    system_prompt: str,
    user_message: str,
    temperature: float,
    model_override: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    message = _openai_chat_completion(messages, temperature, model_override=model_override)
    content = (message or {}).get("content")
    return content.strip() if isinstance(content, str) else ""


# ---------- LLM 저수준 호출: Groq(OpenAI 호환) ----------
def _groq_chat_completion(
    messages: list[dict],
    temperature: float,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    model_override: str | None = None,
) -> dict | None:
    settings = get_settings()
    if not getattr(settings, "groq_api_key", None):
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }
    body = {
        "model": model_override or settings.groq_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

    max_attempts = 3
    resp = None
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("Groq API error %s: %s", status, exc.response.text)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return None
        except httpx.RequestError as exc:
            logger.error("Groq request error: %s", exc)
            if attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return None

    if resp is None:
        return None

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    return choices[0].get("message")


def _call_groq(
    system_prompt: str,
    user_message: str,
    temperature: float,
    model_override: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    message = _groq_chat_completion(messages, temperature, model_override=model_override)
    content = (message or {}).get("content")
    return content.strip() if isinstance(content, str) else ""


# ---------- LLM Failover 래퍼 ----------
def call_llm_with_failover(
    system_prompt: str,
    user_message: str,
    temperature: float,
    primary_override: str | None = None,
    model_override: str | None = None,
) -> str:
    settings = get_settings()
    # gemini 제거, 기본값 openai
    primary = (primary_override or getattr(settings, "primary_llm", "openai") or "openai").lower()

    def use_groq() -> str:
        return _call_groq(system_prompt, user_message, temperature, model_override=model_override)

    def use_openai() -> str:
        return _call_openai(system_prompt, user_message, temperature, model_override=model_override)

    # Gemini 관련 로직 제거, OpenAI -> Groq 순서 위주로 구성
    if primary == "groq":
        result = use_groq()
        if result:
            logger.info("LLM provider=groq")
            return result
        result = use_openai()
        if result:
            logger.info("LLM provider=openai")
            return result
    else:
        # primary가 openai 또는 기타 등등인 경우 기본적으로 OpenAI 우선
        result = use_openai()
        if result:
            logger.info("LLM provider=openai")
            return result
        result = use_groq()
        if result:
            logger.info("LLM provider=groq")
            return result

    logger.warning("LLM failed: openai/groq returned empty")
    return ""


def call_llm_with_tools(
    system_prompt: str,
    user_message: str,
    tool_context: ToolContext | None,
    temperature: float = 0.2,
) -> tuple[str, bool]:
    settings = get_settings()
    primary = (getattr(settings, "primary_llm", "openai") or "openai").lower()
    tools = get_tool_definitions()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    request_id = tool_context.request_id if tool_context else ""
    max_tool_calls = 3
    tool_timeout_seconds = 5.0
    strict_reply_tools = {"wait_status", "doctor_list"}
    total_tool_calls = 0

    def tool_fallback_reply() -> str:
        return (
            "현재 시스템에서 확인이 어렵습니다. "
            "예약 번호나 연락처를 알려주시거나 접수창구로 문의해 주세요."
        )

    def auth_fallback_reply() -> str:
        return "본인 확인 후에만 해당 정보를 확인할 수 있습니다. 인증 정보를 제공해 주세요."

    def strict_tool_reply(name: str, result: dict) -> str | None:
        if name in strict_reply_tools:
            reply_text = result.get("reply_text")
            if isinstance(reply_text, str) and reply_text:
                return reply_text
        return None

    def call_primary(msgs: list[dict]) -> dict | None:
        if primary == "openai":
            message = _openai_chat_completion(msgs, temperature, tools=tools)
            if message:
                return message
            return _groq_chat_completion(msgs, temperature, tools=tools)
        if primary == "groq":
            message = _groq_chat_completion(msgs, temperature, tools=tools)
            if message:
                return message
            return _openai_chat_completion(msgs, temperature, tools=tools)
        logger.warning("primary_llm=%s not supported; using openai then groq", primary)
        message = _openai_chat_completion(msgs, temperature, tools=tools)
        if message:
            return message
        return _groq_chat_completion(msgs, temperature, tools=tools)

    for _ in range(2):
        response_message = call_primary(messages)
        if not response_message:
            return "", False

        tool_calls = response_message.get("tool_calls") or []
        content = response_message.get("content")
        if not tool_calls:
            return (content.strip() if isinstance(content, str) else ""), False

        messages.append(
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            }
        )

        for idx, tool_call in enumerate(tool_calls):
            if total_tool_calls >= max_tool_calls:
                logger.warning("tool call limit reached: request_id=%s total=%s", request_id, total_tool_calls)
                return tool_fallback_reply(), False
            total_tool_calls += 1

            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            start_time = time.monotonic()
            try:
                result = execute_tool(name, args, tool_context)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.exception("tool call exception: name=%s error=%s", name, exc)
                return tool_fallback_reply(), False
            elapsed = time.monotonic() - start_time
            if elapsed > tool_timeout_seconds:
                logger.warning("tool call timeout: request_id=%s name=%s elapsed=%.2fs", request_id, name, elapsed)
                return tool_fallback_reply(), False
            if not isinstance(result, dict):
                logger.warning(
                    "tool call invalid result: request_id=%s name=%s type=%s",
                    request_id,
                    name,
                    type(result),
                )
                return tool_fallback_reply(), False
            if result.get("status") == "error":
                if result.get("error_code") == "auth_required":
                    logger.warning("tool auth required: request_id=%s name=%s", request_id, name)
                    return auth_fallback_reply(), False
                logger.warning(
                    "tool call error: request_id=%s name=%s message=%s",
                    request_id,
                    name,
                    result.get("message"),
                )
                return tool_fallback_reply(), False

            strict_reply = strict_tool_reply(name, result)
            if strict_reply is not None:
                return strict_reply, True

            tool_call_id = tool_call.get("id") or f"tool_call_{idx}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": format_tool_result(result),
                }
            )

    return "", False


# ---------- RAG + 패턴별 말투 + DB 캐싱 ----------
# LLM orchestrator: called by RAG to apply policy and run tool-calling if needed.
def call_gemini_with_rag(
    query: str,
    retrieved_docs: list,
    tool_context: ToolContext | None = None,
    use_tools: bool = False,
    sources_hash: str | None = None,
) -> str:
    """
    RAG 기반 응답 생성 + 질문 패턴별 톤 조절 + DB 기반 캐싱.
    """

    settings = get_settings()

    # 1) 컨텍스트 합치기 (텍스트만 추출)
    parts: list[str] = []
    for d in retrieved_docs:
        text = None
        if isinstance(d, dict):
            text = (
                d.get("text")
                or d.get("snippet")
                or d.get("chunk")
                or d.get("page_content")
            )
        elif isinstance(d, str):
            text = d
        else:
            text = (
                getattr(d, "text", None)
                or getattr(d, "snippet", None)
                or getattr(d, "page_content", None)
            )

        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    context_raw = " ".join(parts)
    context = format_context(context_raw)

    # ✅ 컨텍스트 길이 제한 (너 settings에 있음)
    max_chars = getattr(settings, "max_context_chars", 1200)
    if len(context) > max_chars:
        context = context[:max_chars]

    # 2) DB 캐시 조회 (tool 사용 시 캐시를 우회)
    context_hash = hash_text(context) if context else ""
    if not use_tools:
        cached = get_cached_response(
            query=query,
            intent="rag",
            cache_scope=CACHE_SCOPE_RAG_CONTEXT,
            rag_index_version=settings.rag_index_version,
            top_k=settings.top_k,
            prompt_version=settings.prompt_version,
            sources_hash=sources_hash or context_hash or None,
        )
        if cached:
            return clean_response(cached.response)

    # 3) 질문 패턴 분류 (⭐ 핵심 수정: time 과도매칭 방지 + prep 모드 추가)
    SYMPTOM_KEYWORDS = [
        "아파", "통증", "붓", "부었", "열", "두통", "복통",
        "가슴이", "흉통", "두근", "숨이", "호흡", "기침", "가래",
        "어지럽", "쓰러질", "몸살", "오한", "콧물", "인후통", "목 아프",
        "가려움", "발진", "두드러기", "소변", "배뇨", "빈뇨", "혈뇨",
        "구토", "메스꺼움"
    ]

    EMOTION_KEYWORDS = [
        "우울", "불안", "힘들", "상실감", "지치", "불편한 마음", "멘탈",
        "죽고싶", "살기 싫", "포기하고 싶"
    ]

    # ✅ 운영/진료시간 전용 키워드만 (일반 '시간' 제거)
    TIME_KEYWORDS = [
        "진료시간", "진료 시간",
        "운영시간", "운영 시간",
        "접수시간", "접수 시간",
        "오픈", "마감",
        "몇 시까지", "몇시까지", "몇 시부터", "몇시부터",
        "콜센터 운영", "콜센터 시간"
    ]

    HIGH_RISK_SYMPTOM_KEYWORDS = [
        "의식이 없",
        "의식 소실",
        "실신",
        "호흡 곤란",
        "호흡곤란",
        "숨을 못",
        "가슴 통증",
        "흉통",
        "가슴이 쥐어짜",
        "경련",
        "심한 출혈",
        "대량 출혈",
        "토혈",
        "혈변",
        "검은 변",
        "심한 복통",
        "극심한 통증",
        "말이 어눌",
        "편마비",
        "시야가 갑자기",
        "아나필락시스",
        "알레르기 쇼크",
    ]

    def _should_add_emergency_notice(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if "응급" in t or "119" in t:
            return True
        return any(k in t for k in HIGH_RISK_SYMPTOM_KEYWORDS)

    # ✅ 검사 준비/금식 전용 모드 추가
    PREP_KEYWORDS = [
        "금식", "공복", "혈액검사", "피검사", "초음파", "CT", "MRI", "검사 전", "검사전", "준비", "전처치"
    ]

    def detect_mode(text: str) -> str:
        t = (text or "").strip()

        if any(k in t for k in EMOTION_KEYWORDS):
            return "emotional"

        # ✅ 금식/검사 준비는 time보다 먼저 잡아야 함
        if any(k in t for k in PREP_KEYWORDS):
            return "prep"

        if any(k in t for k in SYMPTOM_KEYWORDS):
            return "symptom"

        if any(k in t for k in TIME_KEYWORDS):
            return "time"

        return "info"

    mode = detect_mode(query)

    # 4) 공통 스타일 규칙 + Policy 추가
    emergency_policy = _load_emergency_policy()

    base_style = """
당신은 병원 공식 챗봇입니다. 존댓말(합니다체)로 1~3문장 이내 핵심만 간결히 답하세요.
추측성 발언이나 불필요한 서두/미사여구를 배제하고, 질문의 핵심 정보(시간/위치/방법 등)를 두괄식으로 제시하세요.
"""

    # ✅ 모드별 추가 규칙 (간소화)
    if mode == "time":
        extra_rule = """
질문 주제: 병원 운영/진료시간.
컨텍스트의 운영시간 정보를 최우선으로 답하고, 없으면 대표번호 문의를 안내하세요.
"""
    elif mode == "prep":
        extra_rule = """
질문 주제: 검사 준비/금식.
컨텍스트의 금식 시간/주의사항을 정확히 전달하세요. 운영시간은 질문과 무관하면 생략하세요.
"""
    elif mode == "symptom":
        if _should_add_emergency_notice(query):
            extra_rule = """
질문 주제: 증상 상담.
증상 설명 및 진료과 안내 위주로 답하세요.
응급(호흡곤란/심한 통증 등) 의심 시: "증상 악화 시 응급실 방문이 필요할 수 있습니다." 문구를 끝에 추가.
"""
        else:
            extra_rule = """
질문 주제: 증상 상담.
증상 설명 및 진료과 안내 위주로 답하세요. 불필요한 응급실 안내는 생략하세요.
"""
    elif mode == "emotional":
        extra_rule = """
질문 주제: 감정/심리.
짧고 담백하게 공감하되, 전문 상담/진료 가능성을 열어두는 정도로만 안내하세요.
"""
    else:  # info
        extra_rule = """
질문 주제: 병원 이용/예약/위치/정보.
컨텍스트의 사실 정보(숫자/전화번호)를 정확하되 건조하게 전달하세요.
"""

    tool_rule = ""
    if use_tools:
        tool_rule = """
[Tool 사용 규칙]
1. 실시간 정보(예약/대기/이력)는 반드시 Tool을 호출해 확인. 추측 금지.
2. 의사 목록 요청 시 doctor_list 호출. (추천 금지)
3. 예약 생성/변경 시 doctor_name/id가 있으면 필히 전달.
4. Tool 결과의 reply_text가 있다면 그대로 답변에 사용.
"""

    # system prompt = (응급 정책) + base_style + extra_rule
    if emergency_policy:
        system_prompt = f"{emergency_policy}\n\n{base_style}\n{extra_rule}\n{tool_rule}".strip()
    else:
        system_prompt = f"{base_style}\n{extra_rule}\n{tool_rule}".strip()

    context_block = ""
    if tool_context and (tool_context.session_id or tool_context.metadata):
        safe_metadata = sanitize_metadata_for_prompt(tool_context.metadata or {})
        context_block = f"""
[요청 컨텍스트]
session_id: {tool_context.session_id or ""}
metadata: {json.dumps(safe_metadata, ensure_ascii=False, default=str)}
"""

    user_message = f"""
{context_block}
[사용자 질문]
{query}

[참고용 컨텍스트]
다음 내용은 사용자의 질문에 답하기 위한 참고 자료입니다.
이 내용을 그대로 복사하지 말고, 의미를 유지하면서 다른 표현으로 정리하여 답변하십시오.
숫자/시간/전화번호 같은 핵심 정보는 누락하지 마세요.

--- context start ---
{context}
--- context end ---
"""

    # 5) LLM 호출 (tool 사용 시 tool loop)
    if use_tools:
        raw_reply, is_tool_reply = call_llm_with_tools(
            system_prompt,
            user_message,
            tool_context,
            temperature=0.2,
        )
    else:
        raw_reply = call_llm_with_failover(system_prompt, user_message, temperature=0.2)
        is_tool_reply = False
    if not raw_reply:
        return "현재 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주시기 바랍니다."

    if is_tool_reply:
        return ToolReply(raw_reply)

    final_reply = clean_response(raw_reply)

    # 6) DB 캐시 저장 (tool 사용 시 캐시를 우회)
    if not use_tools:
        save_cache_response(
            query=query,
            intent="rag",
            cache_scope=CACHE_SCOPE_RAG_CONTEXT,
            rag_index_version=settings.rag_index_version,
            top_k=settings.top_k,
            prompt_version=settings.prompt_version,
            response=final_reply,
            context_text=context,
            context_hash=context_hash,
            sources_hash=sources_hash or context_hash,
        )

    return final_reply
