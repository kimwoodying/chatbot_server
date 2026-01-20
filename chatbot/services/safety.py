from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SafetyResult:
    category: str
    reply: str


SAFETY_PATTERNS = {
    "self_harm": [
        r"죽고\s*싶",
        r"자살",
        r"자해",
        r"해치고\s*싶",
        r"살기\s*싫",
        r"끝내고\s*싶",
    ],
    "violence": [
        r"폭력",
        r"위협",
        r"살해",
        r"폭행",
        r"칼",
        r"총",
    ],
    "abuse": [
        r"학대",
        r"가정폭력",
        r"아동\s*학대",
        r"노인\s*학대",
    ],
    "overdose": [
        r"과다\s*복용",
        r"약물\s*중독",
        r"약을\s*너무",
        r"독성",
    ],
    "pregnancy_highrisk": [
        r"임신.*(출혈|심한\s*통증|구토|실신)",
        r"임산부.*(응급|위급|출혈)",
    ],
    "legal_diagnosis": [
        r"진단서",
        r"법적",
        r"소송",
        r"책임",
    ],
}


SAFETY_TEMPLATES = {
    "self_harm": (
        "현재 안전이 가장 중요합니다. 지금 위험하거나 혼자 감당하기 어렵다면 119나 가까운 응급실로 도움을 요청해 주세요."
    ),
    "violence": (
        "위험하거나 폭력 상황이라면 즉시 안전한 곳으로 이동하고 112 또는 119에 도움을 요청해 주세요."
    ),
    "abuse": (
        "학대나 폭력 위험이 의심되면 즉시 안전을 확보하고 112 또는 119에 도움을 요청해 주세요."
    ),
    "overdose": (
        "약물 과다복용이나 중독이 의심되면 즉시 119에 연락하거나 가까운 응급실을 방문해 주세요."
    ),
    "pregnancy_highrisk": (
        "임신 중 심한 통증이나 출혈 등 위험 증상이 있다면 즉시 119에 연락하거나 응급실로 이동해 주세요."
    ),
    "legal_diagnosis": (
        "법적 판단이나 진단이 필요한 경우에는 진료를 통해 확인이 필요합니다. 원하시면 접수 방법을 안내해 드리겠습니다."
    ),
}


def detect_safety_category(query: str) -> Optional[str]:
    q = (query or "").lower()
    if not q:
        return None
    for category, patterns in SAFETY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return category
    return None


def build_safety_response(query: str) -> Optional[SafetyResult]:
    category = detect_safety_category(query)
    if not category:
        return None
    reply = SAFETY_TEMPLATES.get(category)
    if not reply:
        return None
    return SafetyResult(category=category, reply=reply)
