import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# ===== 설정 =====
# TL/2.답변 폴더를 가리키도록 지정하세요.
# 예: r"D:\aihub\71762\Training\02.라벨링데이터\TL\2.답변"
INPUT_ROOT = Path(r"C:\Users\401\Downloads\AI허브\120.초거대AI 사전학습용 헬스케어 질의응답 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL\2.답변")

# 출력 파일 (JSONL)
OUTPUT_JSONL = Path("aihub_71762_rag_docs.jsonl")

# 섹션 우선순위(문서 합칠 때 이 순서대로 정렬)
SECTION_ORDER = ["증상", "정의", "원인", "진단", "치료", "예방"]

# 의료 리스크 완화 문구(원하면 끄거나 문구 변경)
SAFETY_FOOTER = (
    "※ 본 정보는 일반적인 의료 정보이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
)

# 너무 짧은 문서는 제외(노이즈 제거)
MIN_TEXT_LEN = 80


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def extract_text_from_json(obj: Any) -> str:
    """
    AI Hub JSON 포맷이 파일마다 약간 달라질 수 있어 key 후보를 순서대로 탐색.
    가능한 한 '답변 본문'만 추출한다.
    """
    if obj is None:
        return ""

    # 1) 흔한 케이스: dict
    if isinstance(obj, dict):
        # 가장 자주 쓰이는 키 후보들
        key_candidates = [
            "answer", "Answer", "A", "a",
            "response", "Response",
            "text", "Text",
            "content", "Content",
            "label", "Label",
        ]
        for k in key_candidates:
            if k in obj:
                v = obj.get(k)
                # nested 구조일 수 있음
                if isinstance(v, (dict, list)):
                    t = extract_text_from_json(v)
                    if t.strip():
                        return t
                else:
                    t = _to_str(v).strip()
                    if t:
                        return t

        # 2) dict 내부 값들 중 문자열이 길게 있는 것을 고르는 fallback
        longest = ""
        for v in obj.values():
            if isinstance(v, str) and len(v.strip()) > len(longest):
                longest = v.strip()
        if longest:
            return longest

        # 3) 그래도 없으면 하위 탐색
        for v in obj.values():
            t = extract_text_from_json(v)
            if t.strip():
                return t
        return ""

    # 2) list면 내부 요소를 이어 붙임
    if isinstance(obj, list):
        parts = []
        for it in obj:
            t = extract_text_from_json(it).strip()
            if t:
                parts.append(t)
        return "\n".join(parts).strip()

    # 3) 그 외 타입
    return _to_str(obj).strip()


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 줄바꿈 3개 이상은 2개로 축소
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 양 끝 공백 제거
    return s.strip()


def get_section_sort_key(section: str) -> int:
    try:
        return SECTION_ORDER.index(section)
    except ValueError:
        return len(SECTION_ORDER) + 100  # 알 수 없는 섹션은 뒤로


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        # 임시파일/숨김파일 방지
        if p.name.startswith("~") or p.name.startswith("."):
            continue
        yield p


def parse_path_meta(file_path: Path, input_root: Path) -> Optional[Tuple[str, str, str]]:
    """
    기대 경로: <root>/<대분류>/<질환>/<섹션>/파일.json
    return: (category, disease, section)
    """
    rel = file_path.relative_to(input_root)
    parts = rel.parts
    if len(parts) < 4:
        return None
    category, disease, section = parts[0], parts[1], parts[2]
    return category, disease, section


def build_doc(category: str, disease: str, section_to_texts: Dict[str, List[str]]) -> str:
    # 섹션별 텍스트 합치기
    sections_sorted = sorted(section_to_texts.keys(), key=get_section_sort_key)
    body_parts = [f"질환명: {disease}", f"질환 분류: {category}", ""]
    for sec in sections_sorted:
        texts = [t for t in section_to_texts[sec] if t.strip()]
        if not texts:
            continue
        merged = "\n".join(texts).strip()
        merged = normalize_whitespace(merged)
        body_parts.append(f"[{sec}]")
        body_parts.append(merged)
        body_parts.append("")  # blank line

    if SAFETY_FOOTER:
        body_parts.append(SAFETY_FOOTER)

    return normalize_whitespace("\n".join(body_parts))


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT 경로가 존재하지 않습니다: {INPUT_ROOT}")

    # (category, disease) -> section -> [texts]
    bucket: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    # 디버깅용: 파싱 실패 파일
    failed_files: List[str] = []

    for fp in iter_json_files(INPUT_ROOT):
        meta = parse_path_meta(fp, INPUT_ROOT)
        if meta is None:
            continue
        category, disease, section = meta

        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            # 일부 데이터는 cp949 같은 경우가 있어 fallback
            try:
                data = json.loads(fp.read_text(encoding="cp949"))
            except Exception:
                failed_files.append(str(fp))
                continue
        except Exception:
            failed_files.append(str(fp))
            continue

        text = extract_text_from_json(data)
        text = normalize_whitespace(text)

        # 너무 짧은/의미 없는 건 스킵(원하면 MIN_TEXT_LEN 조정)
        if len(text) < 10:
            continue

        bucket[(category, disease)][section].append(text)

    # JSONL로 쓰기
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for (category, disease), section_map in sorted(bucket.items()):
            doc_text = build_doc(category, disease, section_map)
            if len(doc_text) < MIN_TEXT_LEN:
                continue

            doc = {
                "id": f"{category}/{disease}",
                "text": doc_text,
                "metadata": {
                    "category": category,
                    "disease": disease,
                    "sections": sorted(section_map.keys(), key=get_section_sort_key),
                    "source": "AIHub_71762_TL",
                    "path": str(INPUT_ROOT),
                },
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] JSONL 생성 완료: {OUTPUT_JSONL} (문서 수: {written})")
    if failed_files:
        print(f"[WARN] JSON 파싱 실패 파일 {len(failed_files)}개 (처음 20개):")
        for x in failed_files[:20]:
            print(" -", x)


if __name__ == "__main__":
    main()
