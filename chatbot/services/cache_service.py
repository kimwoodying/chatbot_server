# chatbot/services/cache_service.py
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import timedelta
from typing import Optional

from django.utils import timezone

from chatbot.config import get_settings
from chatbot.models import ChatCache

logger = logging.getLogger(__name__)

CACHE_SCOPE_QUERY_ONLY = "query_only"
CACHE_SCOPE_RAG_CONTEXT = "rag_context"

DYNAMIC_KEYWORDS = [
    "오늘",
    "지금",
    "현재",
    "실시간",
    "금일",
    "방금",
    "마감",
    "접수",
    "대기",
    "순번",
    "예약",
    "가능",
    "남은",
    "변경",
]


def normalize_query(query: str) -> str:
    normalized = (query or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_cache_key(
    intent: str,
    normalized_query: str,
    rag_index_version: str,
    top_k: int,
    prompt_version: str,
    cache_scope: str,
    sources_hash: str | None = None,
) -> str:
    payload = {
        "intent": intent or "",
        "normalized_query": normalized_query,
        "rag_index_version": rag_index_version,
        "top_k": top_k,
        "prompt_version": prompt_version,
        "cache_scope": cache_scope,
    }
    if sources_hash:
        payload["sources_hash"] = sources_hash
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def classify_cache_ttl_seconds(query: str, intent: str, cache_scope: str) -> Optional[int]:
    settings = get_settings()
    if intent == "tool":
        return None
    if cache_scope == CACHE_SCOPE_QUERY_ONLY and intent == "rag":
        return settings.cache_ttl_default_seconds

    q = normalize_query(query)
    if any(k in q for k in DYNAMIC_KEYWORDS):
        return settings.cache_ttl_dynamic_seconds
    if intent in {"static", "safety"}:
        return settings.cache_ttl_static_seconds
    return settings.cache_ttl_default_seconds


def make_query_hash(cache_key: str) -> str:
    return hash_text(cache_key)


def get_cached_response(
    *,
    query: str,
    intent: str,
    cache_scope: str,
    rag_index_version: str,
    top_k: int,
    prompt_version: str,
    sources_hash: str | None = None,
) -> ChatCache | None:
    normalized_query = normalize_query(query)
    cache_key = build_cache_key(
        intent=intent,
        normalized_query=normalized_query,
        rag_index_version=rag_index_version,
        top_k=top_k,
        prompt_version=prompt_version,
        cache_scope=cache_scope,
        sources_hash=sources_hash,
    )
    qh = make_query_hash(cache_key)
    cache = ChatCache.objects.filter(query_hash=qh).first()
    if not cache:
        return None
    if cache.expires_at and cache.expires_at <= timezone.now():
        cache.delete()
        return None
    cache.hit_count += 1
    cache.save(update_fields=["hit_count"])
    logger.info("cache hit: scope=%s intent=%s hash=%s", cache_scope, intent, cache.query_hash)
    return cache


def save_cache_response(
    *,
    query: str,
    intent: str,
    cache_scope: str,
    rag_index_version: str,
    top_k: int,
    prompt_version: str,
    response: str,
    context_text: str = "",
    context_hash: str = "",
    sources_hash: str = "",
    sources: list[dict] | None = None,
) -> ChatCache | None:
    ttl_seconds = classify_cache_ttl_seconds(query, intent, cache_scope)
    if ttl_seconds is None:
        return None

    normalized_query = normalize_query(query)
    cache_key = build_cache_key(
        intent=intent,
        normalized_query=normalized_query,
        rag_index_version=rag_index_version,
        top_k=top_k,
        prompt_version=prompt_version,
        cache_scope=cache_scope,
        sources_hash=sources_hash or None,
    )
    qh = make_query_hash(cache_key)
    expires_at = timezone.now() + timedelta(seconds=ttl_seconds)

    cache, created = ChatCache.objects.get_or_create(
        query_hash=qh,
        defaults={
            "cache_key": cache_key,
            "intent": intent,
            "cache_scope": cache_scope,
            "normalized_query": normalized_query,
            "rag_index_version": rag_index_version,
            "top_k": top_k,
            "prompt_version": prompt_version,
            "query": query,
            "context": context_text or "",
            "context_hash": context_hash or "",
            "sources_hash": sources_hash or "",
            "response": response,
            "sources": sources or None,
            "expires_at": expires_at,
            "hit_count": 1,
        },
    )
    if not created:
        cache.response = response
        cache.context = context_text or ""
        cache.context_hash = context_hash or ""
        cache.sources_hash = sources_hash or ""
        cache.sources = sources or None
        cache.expires_at = expires_at
        cache.hit_count += 1
        cache.save(
            update_fields=[
                "response",
                "context",
                "context_hash",
                "sources_hash",
                "sources",
                "expires_at",
                "hit_count",
            ]
        )
    logger.info("cache saved: scope=%s intent=%s hash=%s", cache_scope, intent, qh)
    return cache


def clear_cache() -> int:
    deleted_count, _ = ChatCache.objects.all().delete()
    return deleted_count
