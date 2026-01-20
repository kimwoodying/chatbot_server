# chatbot/services/flows.py
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

from chatbot.services.tooling import ToolContext, execute_tool
from chatbot.services.intents.keywords import MEDICAL_HISTORY_CUES

logger = logging.getLogger(__name__)

# SYMPTOM GUIDE LOGIC
SYMPTOM_GUIDE_PATH = Path(__file__).resolve().parents[1] / "data" / "symptom_guide.json"

def _symptom_guide_cache_key() -> str:
    if not SYMPTOM_GUIDE_PATH.exists():
        return "missing"
    try:
        return str(SYMPTOM_GUIDE_PATH.stat().st_mtime_ns)
    except OSError:
        return "error"

@lru_cache(maxsize=4)
def _load_symptom_guide(cache_key: str) -> List[Dict[str, Any]]:
    _ = cache_key
    if not SYMPTOM_GUIDE_PATH.exists():
        return []
    try:
        with SYMPTOM_GUIDE_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        logger.warning("symptom guide load failed: %s", exc)
        return []

    items = raw.get("items", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keywords = [
            kw.strip()
            for kw in item.get("keywords", [])
            if isinstance(kw, str) and kw.strip()
        ]
        department = item.get("department")
        possible = [
            cause.strip()
            for cause in item.get("possible_causes", [])
            if isinstance(cause, str) and cause.strip()
        ]
        summary = item.get("summary")
        if not keywords or not isinstance(department, str) or not department.strip():
            continue
        cleaned.append(
            {
                "keywords": keywords,
                "department": department.strip(),
                "possible_causes": possible,
                "summary": summary.strip() if isinstance(summary, str) else "",
            }
        )
    return cleaned

def match_symptom_guide(query: str) -> Dict[str, Any] | None:
    if not query:
        return None
    best_entry: Dict[str, Any] | None = None
    best_score = 0
    for entry in _load_symptom_guide(_symptom_guide_cache_key()):
        score = sum(1 for kw in entry.get("keywords", []) if kw in query)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry

# MEDICAL HISTORY LOGIC
def handle_medical_history(
    query: str,
    session_id: str | None,
    metadata: Dict[str, Any] | None,
) -> dict | None:
    """
    진료 내역 조회 로직 분리.
    """
    if not query:
        return None
    if not any(cue in query for cue in MEDICAL_HISTORY_CUES):
        return None
    
    # ToolContext construction
    tool_context = ToolContext(
        session_id=session_id,
        metadata=metadata,
        user_id=metadata.get("user_id") if metadata else None
    )

    result = execute_tool("medical_history", {}, tool_context)
    if isinstance(result, dict) and result.get("reply_text"):
        payload = {"reply": result["reply_text"], "sources": []}
        if result.get("table"):
            payload["table"] = result["table"]
        return payload
    if isinstance(result, dict) and result.get("status") == "not_found":
        return {
            "reply": "현재 진료내역이 없습니다. 원하시면 예약을 도와드리겠습니다.",
            "sources": [],
        }
    if isinstance(result, dict) and result.get("status") == "error":
        return {
            "reply": "진료내역을 확인하려면 환자 정보를 알려주세요.",
            "sources": [],
        }
    return {
        "reply": "진료내역을 확인하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        "sources": [],
    }
