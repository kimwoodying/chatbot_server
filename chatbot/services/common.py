# chatbot/services/common.py
from __future__ import annotations

import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Constants
DEFAULT_GREETING_REPLY = "안녕하세요! 건양대학교병원 챗봇입니다. 무엇을 도와드릴까요?"
AUTH_REQUIRED_REPLY = "로그인 후 이용해 주세요, 전화 문의는 대표번호 1577-3330으로 부탁드립니다."

AUTH_METADATA_KEYS = {
    "patient_id",
    "patient_identifier",
    "patient_phone",
    "account_id",
    "patient_pk",
    "auth_user_id",
    "user_id",
}

# Marker type to bypass post-processing for strict tool replies.
class ToolReply(str):
    pass

def format_context(text: str) -> str:
    """컨텍스트에서 불필요한 포맷 제거."""
    if not text:
        return ""

    cleaned = text
    cleaned = re.sub(r"(참고자료|출처)", "", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)      # 마크다운 제목
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)     # 리스트 기호
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def clean_response(text: str) -> str:
    """
    LLM 응답을 프론트로 보내기 전 마지막 정리.
    - ToolReply 인스턴스면 그대로 반환.
    - 마크다운(굵게, 밑줄, 헤더, 리스트 등) 제거.
    - 문장 끝 마침표 보정.
    """
    if isinstance(text, ToolReply):
        return str(text)
    
    if not text:
        return "" # gemini_client에서는 빈 문자열 리턴하고 상위에서 에러메시지 처리하는 경우도 있음. 
                  # 하지만 rag.py에서는 에러메시지 리턴함. 
                  # gemini_client.py: line 47 return ""
                  # rag.py: line 95 return "죄송합니다..."
                  # 통합을 위해 빈 문자열을 리턴하고 호출처에서 처리? 
                  # rag.py의 clean_response는 최종 응답 직전.
                  # gemini_client의 clean_response는 LLM 응답 후처리.
    
    # 만약 rag.py 의 로직을 따른다면:
    if not text:
        return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."

    # Markdown Bold/Italic 제거
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    
    # Markdown Header 제거 (# Title -> Title)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    
    # Markdown List 제거 (- Item -> Item)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # 링크 포맷 제거 ([Text](Url) -> Text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    text = text.strip()

    if text and text[-1] not in {".", "!", "?", "다", "~"}:
        text += "."

    return text
