# chatbot/config.py
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py location: chat-django/chatbot/config.py
BASE_DIR = Path(__file__).resolve().parent  # chatbot package root


class Settings(BaseSettings):
    """
    Global application settings loaded from .env or environment variables.
    RAG / Gemini / Groq에서 공통으로 사용하는 설정들.
    """

    # ---- 🔑 API Keys ----
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    holiday_api_key: str | None = None
    holiday_api_base_url: str = (
        "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    )
    holiday_api_timeout_seconds: int = 5

    openai_model: str = "gpt-4o-mini"
    groq_model: str = "llama-3.1-8b-instant"

    # ---- 🧠 Embedding model ----
    embedding_model: str = "jhgan/ko-sroberta-multitask"

    # ---- 📂 Data / Vector store paths ----
    # data directory: chat-django/chatbot/data
    data_dir: Path = BASE_DIR / "data"

    # FAISS 인덱스 / 메타데이터 경로
    faiss_index_path: Path = data_dir / "faiss.index"
    metadata_path: Path = data_dir / "metadata.json"

    # ---- 🔍 RAG 검색 / 성능 옵션 ----
    top_k: int = 3
    max_context_chars: int = 1200

    # RAG 인덱스/프롬프트 버전 (캐시 키에 포함)
    rag_index_version: str = "v1"
    prompt_version: str = "v1"

    # ---- 캐시 TTL(초) ----
    cache_ttl_static_seconds: int = 60 * 60 * 24 * 7   # 7일
    cache_ttl_dynamic_seconds: int = 60 * 5           # 5분
    cache_ttl_default_seconds: int = 60 * 60 * 24     # 1일

    # ---- Tool auth ----
    tool_auth_required: bool = True

    # ---- Intent routing LLM (비용/성능 최적화용) ----
    intent_llm_provider: str | None = None
    intent_llm_model: str | None = None

    # 어떤 LLM을 1순위로 쓸지
    primary_llm: str = "openai"  # openai / gemini / groq

    # ---- Optional ----
    database_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
