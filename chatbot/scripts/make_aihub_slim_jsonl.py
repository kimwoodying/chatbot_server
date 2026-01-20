import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ===== 설정 =====
INPUT_JSONL = Path(r"C:\Users\401\Desktop\flutter\chat-django\chatbot\aihub_71762_rag_docs.jsonl")     # (너가 이미 만든 질환단위 문서)
OUTPUT_JSONL = Path(r"C:\Users\401\Desktop\flutter\chat-django\chatbot\chatbot\aihub_71762_rag_slim.jsonl")    # (새로 만들 슬림 문서)

# RAG에 남길 섹션(기본: 증상/정의/원인만)
KEEP_SECTIONS = ["증상", "정의", "원인"]

# 필요하면 True로 바꾸면 포함됨(길어지니까 기본 False 추천)
INCLUDE_DIAGNOSIS = False   # 진단
INCLUDE_TREATMENT = False   # 치료
INCLUDE_PREVENTION = False  # 예방

MAX_SECTION_CHARS = 800     # 섹션별로 너무 길면 앞부분만 사용(노이즈/비용 방지)
MAX_DOC_CHARS = 1300        # 한 문서 총 길이 제한(추가 안전장치)
MIN_DOC_CHARS = 120         # 너무 짧으면 제외

SAFETY_FOOTER = "※ 본 정보는 일반적인 의료 정보이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."

SECTION_PATTERN = re.compile(r"^\[(.+?)\]\s*$", re.MULTILINE)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def parse_sections(full_text: str) -> Dict[str, str]:
    """
    너 문서 포맷:
      질환명: ...
      질환 분류: ...

      [증상]
      ...
      [정의]
      ...

    위 구조를 섹션별로 분해해서 dict로 반환
    """
    lines = full_text.splitlines()
    sections: Dict[str, List[str]] = {}
    current = None

    for line in lines:
        m = re.match(r"^\[(.+?)\]\s*$", line.strip())
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def build_slim_text(category: str, disease: str, sections: Dict[str, str]) -> Optional[str]:
    keep = list(KEEP_SECTIONS)

    if INCLUDE_DIAGNOSIS:
        keep.append("진단")
    if INCLUDE_TREATMENT:
        keep.append("치료")
    if INCLUDE_PREVENTION:
        keep.append("예방")

    parts = [f"질환명: {disease}", f"질환 분류: {category}", ""]

    total_len = 0
    kept_any = False

    for sec in keep:
        body = sections.get(sec, "").strip()
        if not body:
            continue
        kept_any = True
        body = _clip(body, MAX_SECTION_CHARS)

        block = f"[{sec}]\n{body}".strip()
        # 총 길이 제한
        if total_len + len(block) > MAX_DOC_CHARS:
            remaining = max(0, MAX_DOC_CHARS - total_len - len(f"[{sec}]\n") - 3)
            if remaining > 50:
                block = f"[{sec}]\n{_clip(body, remaining)}"
                parts.append(block)
                total_len += len(block)
            break

        parts.append(block)
        parts.append("")
        total_len += len(block) + 2

    if not kept_any:
        return None

    parts.append(SAFETY_FOOTER)
    text = "\n".join(parts).strip()

    if len(text) < MIN_DOC_CHARS:
        return None

    return text


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {INPUT_JSONL.resolve()}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            obj = json.loads(line)
            doc_id = obj.get("id", f"doc_{total}")
            full_text = obj.get("text", "")
            meta = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}

            category = meta.get("category") or "unknown"
            disease = meta.get("disease") or (doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id)

            sections = parse_sections(full_text)
            slim_text = build_slim_text(category, disease, sections)
            if not slim_text:
                skipped += 1
                continue

            slim_doc = {
                "id": doc_id,
                "text": slim_text,
                "metadata": {
                    **meta,
                    "category": category,
                    "disease": disease,
                    "doc_type": "aihub_71762_slim",
                    "kept_sections": KEEP_SECTIONS
                    + (["진단"] if INCLUDE_DIAGNOSIS else [])
                    + (["치료"] if INCLUDE_TREATMENT else [])
                    + (["예방"] if INCLUDE_PREVENTION else []),
                    "source": meta.get("source") or "AIHub_71762_TL",
                }
            }
            fout.write(json.dumps(slim_doc, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] slim JSONL 생성 완료: {OUTPUT_JSONL} / total={total} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
