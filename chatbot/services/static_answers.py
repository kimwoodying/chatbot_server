from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StaticAnswerResult:
    reply: str
    sources: list[dict]
    name: str


def extract_context_text(meta: Dict[str, Any]) -> str:
    def _clean(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    for key in ("text", "snippet", "chunk", "page_content", "content"):
        text = _clean(meta.get(key))
        if text:
            return text
    return ""


def extract_contact_numbers(text: str) -> Dict[str, str]:
    normalized = (text or "").replace("*", "")
    patterns = {
        "대표번호": r"(?:대표번호|대표\s*전화|대표전화)\s*[:：]?\s*(\d{2,4}-\d{3,4}-\d{4}|\d{4}-\d{4})",
        "응급실": r"응급실\s*[:：]?\s*(\d{2,4}-\d{3,4}-\d{4}|\d{4}-\d{4})",
        "콜센터": r"콜센터\s*[:：]?\s*(\d{2,4}-\d{3,4}-\d{4}|\d{4}-\d{4})",
    }
    found: Dict[str, str] = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, normalized)
        if match:
            found[label] = match.group(1)
    return found


def extract_cancer_centers(text: str) -> List[str]:
    centers = ["위암", "대장암", "간암", "유방암", "폐암"]
    found: List[str] = []
    for name in centers:
        if name in text:
            found.append(name)
    return found


HOSPITAL_INFO_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "hospital_info.txt"
PARKING_INFO_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "parking_info.txt"


@lru_cache(maxsize=1)
def _load_hospital_info_text() -> str:
    try:
        if HOSPITAL_INFO_PATH.exists():
            return HOSPITAL_INFO_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


@lru_cache(maxsize=1)
def _load_parking_info_text() -> str:
    try:
        if PARKING_INFO_PATH.exists():
            return PARKING_INFO_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def extract_location_line(text: str) -> str:
    if not text:
        return ""
    for raw in text.splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line or line in {"위치", "주소"}:
            continue
        if any(k in line for k in ["지하철", "버스", "하차", "인근", "이용"]):
            continue
        has_region = "시" in line and ("구" in line or "군" in line)
        has_street = any(k in line for k in ["로", "길", "동"])
        has_digit = any(ch.isdigit() for ch in line)
        if (has_region and has_street) or (has_digit and has_street):
            return line
    return ""


def extract_operating_hours(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("*", "")
    for raw in normalized.splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        if "외래 진료" in line or "외래진료" in line:
            match = re.search(r"(\d{1,2}:\d{2})\s*~\s*(\d{1,2}:\d{2})", line)
            if match:
                return f"{match.group(1)}~{match.group(2)}"
    return ""


def extract_location_info(metas: List[Dict[str, Any]], contexts_text: List[str]) -> str:
    raw_text = _load_hospital_info_text()
    if raw_text:
        loc = extract_location_line(raw_text)
        if loc:
            return loc

    for meta in metas:
        if not isinstance(meta, dict):
            continue
        section = str(meta.get("section") or "")
        title = str(meta.get("title") or "")
        source_file = str(meta.get("source_file") or "")
        if not any(k in section for k in ["위치", "주소"]) and not any(
            k in title for k in ["위치", "주소"]
        ) and "hospital_info" not in source_file:
            continue
        text = extract_context_text(meta)
        loc = extract_location_line(text)
        if loc:
            return loc
    return ""


def _parse_parking_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = ""
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        if line.startswith("["):
            continue
        if any(k in line for k in ["주차 문의", "주차문의", "주차정산소", "전화 응답"]):
            continue
        if "정산" in line:
            continue
        if "외래" in line and "환자" in line and not any(ch.isdigit() for ch in line):
            current = "외래 환자"
            continue
        if any(k in line for k in ["입원", "퇴원", "수술"]) and not any(
            ch.isdigit() for ch in line
        ):
            current = "입원·퇴원·수술 당일"
            continue
        if "면회객" in line and not any(ch.isdigit() for ch in line):
            current = "면회객"
            continue
        if current:
            sections.setdefault(current, []).append(line)
    return sections


def extract_parking_summary(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("*", "")
    parts: List[str] = []
    if "입차 시간" in normalized:
        parts.append("주차 요금은 입차 시간 기준으로 부과됩니다.")
    sections = _parse_parking_sections(normalized)
    for name in ["외래 환자", "입원·퇴원·수술 당일", "면회객"]:
        details = sections.get(name, [])
        if details:
            detail_text = ", ".join(details)
            parts.append(f"{name}은 {detail_text}입니다.")
    return " ".join(parts).strip()


def extract_parking_settlement(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("*", "")
    phone = ""
    hours = ""
    phone_match = re.search(r"주차정산소[:\s]*([0-9-]+)", normalized)
    if phone_match:
        phone = phone_match.group(1)
    hours_match = re.search(r"(?:전화\s*응답\s*시간|운영\s*시간)[:\s]*([0-9:~\s]+)", normalized)
    if hours_match:
        hours = re.sub(r"\s+", " ", hours_match.group(1)).strip()
        hours = hours.replace("(주말 포함)", "").strip()
    parts: List[str] = []
    if phone:
        parts.append(f"정산은 주차정산소({phone})에서 가능합니다.")
    else:
        parts.append("정산은 주차정산소에서 가능합니다.")
    if hours:
        parts.append(f"운영 시간은 {hours}입니다.")
    return " ".join(parts).strip()


def collect_contact_numbers(metas: List[Dict[str, Any]]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for meta in metas:
        text = extract_context_text(meta)
        if not text:
            continue
        numbers = extract_contact_numbers(text)
        for label, number in numbers.items():
            if label not in found:
                found[label] = number
        if len(found) >= 3:
            break
    return found


def _with_contact_footer(text: str, all_numbers: Dict[str, str]) -> str:
    if not text:
        return text
    rep = all_numbers.get("대표번호")
    if rep:
        return f"{text} 필요하시면 대표번호({rep})로 문의해 주세요."
    return f"{text} 필요하시면 병원 대표번호로 문의해 주세요."


def get_static_answer(
    query: str,
    contexts_text: List[str],
    all_metas: List[Dict[str, Any]],
) -> Optional[StaticAnswerResult]:
    q = query or ""
    all_numbers = collect_contact_numbers(all_metas)

    location_query = any(k in q for k in ["위치", "주소", "어디"])
    time_query = any(k in q for k in ["진료시간", "진료 시간", "운영시간", "운영 시간", "접수시간", "접수 시간"])
    parking_query = any(k in q for k in ["주차", "주차요금", "주차 요금", "주차비", "주차료", "정산", "정산소"])
    settlement_query = any(k in q for k in ["정산", "정산소"])
    contact_query = any(k in q for k in ["대표", "응급", "콜센터", "전화", "연락", "번호"])
    if location_query and not contact_query:
        location_line = extract_location_info(all_metas, contexts_text)
        if location_line:
            return StaticAnswerResult(
                reply=_with_contact_footer(f"병원 위치는 {location_line}입니다.", all_numbers),
                sources=[{"type": "static", "name": "hospital_location"}],
                name="hospital_location",
            )
        return StaticAnswerResult(
            reply="제공된 자료에서 위치 정보를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
            sources=[{"type": "static", "name": "hospital_location"}],
            name="hospital_location",
        )
    if time_query:
        raw_text = _load_hospital_info_text()
        hours = extract_operating_hours(raw_text)
        if hours:
            return StaticAnswerResult(
                reply=_with_contact_footer(f"외래 진료 시간은 {hours}입니다.", all_numbers),
                sources=[{"type": "static", "name": "hospital_hours"}],
                name="hospital_hours",
            )
        return StaticAnswerResult(
            reply="제공된 자료에서 진료시간 정보를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
            sources=[{"type": "static", "name": "hospital_hours"}],
            name="hospital_hours",
        )
    if parking_query:
        parking_text = _load_parking_info_text()
        if settlement_query:
            settlement = extract_parking_settlement(parking_text)
            if settlement:
                return StaticAnswerResult(
                    reply=settlement,
                    sources=[{"type": "static", "name": "parking_settlement"}],
                    name="parking_settlement",
                )
            return StaticAnswerResult(
                reply="정산 안내를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
                sources=[{"type": "static", "name": "parking_settlement"}],
                name="parking_settlement",
            )
        summary = extract_parking_summary(parking_text)
        if summary:
            return StaticAnswerResult(
                reply=_with_contact_footer(summary, all_numbers),
                sources=[{"type": "static", "name": "parking_fees"}],
                name="parking_fees",
            )
        return StaticAnswerResult(
            reply="제공된 자료에서 주차요금 정보를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
            sources=[{"type": "static", "name": "parking_fees"}],
            name="parking_fees",
        )
    if contact_query:
        numbers = extract_contact_numbers("\n".join(contexts_text))
        for label, number in all_numbers.items():
            numbers.setdefault(label, number)

        want_location = location_query
        location_line = extract_location_info(all_metas, contexts_text) if want_location else ""

        want_rep = "대표" in q
        want_er = "응급" in q
        want_call = "콜센터" in q
        if want_rep or want_er or want_call:
            parts = []
            if location_line:
                parts.append(f"위치는 {location_line}입니다.")
            if want_rep and numbers.get("대표번호"):
                parts.append(f"병원 대표번호는 {numbers['대표번호']}입니다.")
            if want_call and numbers.get("콜센터"):
                parts.append(f"콜센터 전화번호는 {numbers['콜센터']}입니다.")
            if want_er and numbers.get("응급실"):
                parts.append(f"응급실 전화번호는 {numbers['응급실']}입니다.")
            if parts:
                return StaticAnswerResult(
                    reply=" ".join(parts) + " 필요하시면 추가로 안내해 드리겠습니다.",
                    sources=[{"type": "static", "name": "contact_numbers"}],
                    name="contact_numbers",
                )
            return StaticAnswerResult(
                reply="현재 제공된 자료에서 요청하신 연락처를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
                sources=[{"type": "static", "name": "contact_numbers"}],
                name="contact_numbers",
            )

        parts = []
        if location_line:
            parts.append(f"위치는 {location_line}입니다.")
        if numbers.get("대표번호"):
            parts.append(f"병원 대표번호는 {numbers['대표번호']}입니다.")
        if numbers.get("콜센터"):
            parts.append(f"콜센터 전화번호는 {numbers['콜센터']}입니다.")
        if numbers.get("응급실"):
            parts.append(f"응급실 전화번호는 {numbers['응급실']}입니다.")
        if parts:
            return StaticAnswerResult(
                reply=" ".join(parts) + " 필요하시면 추가로 안내해 드리겠습니다.",
                sources=[{"type": "static", "name": "contact_numbers"}],
                name="contact_numbers",
            )
        return StaticAnswerResult(
            reply="현재 제공된 자료에서 연락처 정보를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
            sources=[{"type": "static", "name": "contact_numbers"}],
            name="contact_numbers",
        )

    cancer_query = any(k in q for k in ["암센터", "위암", "대장암", "간암", "유방암", "폐암"])
    if cancer_query:
        all_numbers = collect_contact_numbers(all_metas)
        centers = extract_cancer_centers("\n".join(contexts_text))
        if not centers:
            for meta in all_metas:
                text = extract_context_text(meta)
                if not text:
                    continue
                for name in extract_cancer_centers(text):
                    if name not in centers:
                        centers.append(name)
                if len(centers) >= 5:
                    break

        if centers:
            centers_text = ", ".join(centers)
            if all_numbers.get("대표번호"):
                return StaticAnswerResult(
                    reply=(
                        f"병원에서 운영하는 암센터는 {centers_text}입니다. "
                        f"자세한 내용은 대표번호({all_numbers['대표번호']})로 문의해 주세요."
                    ),
                    sources=[{"type": "static", "name": "cancer_centers"}],
                    name="cancer_centers",
                )
            return StaticAnswerResult(
                reply=(
                    f"병원에서 운영하는 암센터는 {centers_text}입니다. "
                    "자세한 내용은 병원 안내 데스크로 문의해 주세요."
                ),
                sources=[{"type": "static", "name": "cancer_centers"}],
                name="cancer_centers",
            )

        if all_numbers.get("대표번호"):
            return StaticAnswerResult(
                reply=(
                    "제공된 자료에서 암센터 정보를 확인하지 못했습니다. "
                    f"대표번호({all_numbers['대표번호']})로 문의해 주세요."
                ),
                sources=[{"type": "static", "name": "cancer_centers"}],
                name="cancer_centers",
            )
        return StaticAnswerResult(
            reply="제공된 자료에서 암센터 정보를 확인하지 못했습니다. 병원 안내 데스크로 문의해 주세요.",
            sources=[{"type": "static", "name": "cancer_centers"}],
            name="cancer_centers",
        )

    return None
