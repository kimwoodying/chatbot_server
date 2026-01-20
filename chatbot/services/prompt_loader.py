# chatbot/services/prompt_loader.py
from __future__ import annotations

from pathlib import Path

from chatbot.config import get_settings


def build_system_prompt() -> str:
    """
    ✅ 응급/안전 규칙(emergency_notice.txt)을 '항상' system prompt에 포함
    ✅ RAG 검색 결과로 응급 문서를 섞지 않도록(이미 ingest에서 제외했음)
    """
    settings = get_settings()

    # 프로젝트 루트 기준 data/raw
    # config.BASE_DIR가 chat-django(루트)면 data/raw는 chat-django/chatbot/data/raw 가 아니라
    # 너 프로젝트 구조에 맞춰 아래 경로를 조정해도 됨.
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    emergency_path = raw_dir / "emergency_notice.txt"

    emergency_text = ""
    if emergency_path.exists():
        emergency_text = emergency_path.read_text(encoding="utf-8").strip()

    # ✅ 여기에 병원 챗봇 공통 규칙을 추가로 붙이면 됨
    base_policy = """
너는 병원 안내 챗봇이다.
- 의료적 판단/확진/진단을 단정하지 않는다.
- 개인정보(주민번호 등 민감정보)는 요구하지 않는다.
- 실시간 정보(대기시간/예약가능시간/순번)는 추정하지 말고 시스템 조회가 필요하다고 안내한다.
"""

    if emergency_text:
        return f"{emergency_text}\n\n{base_policy}".strip()

    return base_policy.strip()
