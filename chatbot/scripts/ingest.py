# chatbot/services/ingest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import faiss
import numpy as np

from chatbot.config import get_settings
from chatbot.services.embeddings import embed_texts


EXCLUDE_FROM_RAG = {
    "emergency_notice.txt",  # ✅ Policy로 고정할 문서라 RAG 인덱싱에서 제외
}

EXCLUDE_JSONL_DOC_TYPES = {"policy", "safety", "notice"}
EXCLUDE_JSONL_TITLES = {
    "응급 상황 안내",
    "병원 기본 안내",
    "진료 예약 안내",
    "검사 전 준비사항",
}
MIN_JSONL_TEXT_LEN = 30
MIN_JSONL_CHUNK_LEN = 50


DOC_TYPE_MAP = {
    "hospital_info.txt": "info",
    "reservation_guide.txt": "reservation",
    "exam_preparation.txt": "prep",
    "departments.txt": "dept",
    "department_mapping.txt": "dept_map",
    "admission_discharge.txt": "admission",
    "parking_info.txt": "parking",
    "outpatient_guide.txt": "outpatient",
    "medical_equipment.txt": "equipment",
    "social_services.txt": "services",
    "cancer_center.txt": "cancer",
}


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if stripped.endswith(":") and len(stripped) <= 40:
        return True
    return False


def _find_section_headers(text: str) -> list[tuple[int, str]]:
    headers: list[tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(True):
        if _is_section_header(line):
            header = line.strip().strip("#").strip("[]").strip()
            if header:
                headers.append((pos, header))
        pos += len(line)
    return headers


def _section_for_offset(headers: list[tuple[int, str]], offset: int) -> str:
    current = ""
    for pos, header in headers:
        if pos <= offset:
            current = header
        else:
            break
    return current


def _strip_frontmatter(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :]).lstrip()
    return text


def _iter_paragraphs(
    text: str,
    detect_headers: bool,
) -> Iterable[tuple[str | None, str]]:
    current_section = ""
    buffer: List[str] = []
    for line in text.splitlines():
        if detect_headers and _is_section_header(line):
            if buffer:
                paragraph = "\n".join(buffer).strip()
                if paragraph:
                    yield current_section or None, paragraph
                buffer = []
            header = line.strip().strip("#").strip("[]").strip()
            if header:
                current_section = header
            continue

        if not line.strip():
            if buffer:
                paragraph = "\n".join(buffer).strip()
                if paragraph:
                    yield current_section or None, paragraph
                buffer = []
            continue

        buffer.append(line.strip())

    if buffer:
        paragraph = "\n".join(buffer).strip()
        if paragraph:
            yield current_section or None, paragraph


def _split_long_paragraph(text: str, max_len: int, overlap: int) -> List[str]:
    if max_len <= 0:
        return [text] if text else []
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if overlap > 0 else end
    return parts


def chunk_text(
    text: str,
    max_len: int = 400,
    overlap: int = 50,
    add_section_prefix: bool = True,
    detect_headers: bool = True,
) -> list[tuple[str, str | None]]:
    text = text.strip()
    if not text:
        return []

    paragraphs = list(_iter_paragraphs(text, detect_headers))
    if not paragraphs:
        return []

    chunks: list[tuple[str, str | None]] = []
    current_parts: List[str] = []
    current_section: str | None = None
    current_len = 0

    def _finalize() -> None:
        nonlocal current_parts, current_section, current_len
        if not current_parts:
            return
        body = "\n\n".join(current_parts).strip()
        if not body:
            current_parts = []
            current_section = None
            current_len = 0
            return
        chunk = f"{current_section}\n{body}" if add_section_prefix and current_section else body
        chunks.append((chunk, current_section))
        current_parts = []
        current_section = None
        current_len = 0

    for section, paragraph in paragraphs:
        if not paragraph:
            continue
        section_value = section or ""
        prefix_len = len(section_value) + 1 if add_section_prefix and section_value else 0
        max_body_len = max_len - prefix_len if max_len > prefix_len else max_len

        if len(paragraph) > max_body_len:
            _finalize()
            for part in _split_long_paragraph(paragraph, max_body_len, overlap):
                part_text = f"{section_value}\n{part}" if add_section_prefix and section_value else part
                if part_text:
                    chunks.append((part_text, section_value or None))
            continue

        if current_parts and current_section != (section_value or None):
            _finalize()

        projected_len = current_len + (2 if current_parts else 0) + len(paragraph)
        current_prefix_len = len(current_section) + 1 if add_section_prefix and current_section else 0
        if current_parts and (projected_len + current_prefix_len) > max_len:
            _finalize()

        if not current_parts:
            current_section = section_value or None
            current_parts = [paragraph]
            current_len = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_len += 2 + len(paragraph)

    _finalize()
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument(
        "--exclude_jsonl_tokens",
        type=str,
        default="",
        help="comma-separated tokens to exclude by doc_id/source",
    )
    parser.add_argument(
        "--include_raw_txt",
        action="store_true",
        help="ingest raw txt files along with jsonl",
    )
    return parser.parse_args()


def _iter_jsonl_docs(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            doc_id = obj.get("id") or f"doc_{line_no}"
            yield {"id": doc_id, "text": text, "metadata": metadata}


def _should_exclude_jsonl(
    doc_meta: Dict[str, Any],
    text: str,
    doc_id: str,
    source: str,
    title: str,
    exclude_tokens: List[str],
) -> bool:
    doc_type = str(doc_meta.get("doc_type", "")).lower()
    category = str(doc_meta.get("category", "")).lower()
    if doc_type in EXCLUDE_JSONL_DOC_TYPES or category in EXCLUDE_JSONL_DOC_TYPES:
        return True
    title_value = (title or "").strip()
    if title_value and title_value in EXCLUDE_JSONL_TITLES:
        return True
    if exclude_tokens:
        haystack = f"{doc_id} {source}".lower()
        if any(token in haystack for token in exclude_tokens):
            return True
    if len(text.strip()) < MIN_JSONL_TEXT_LEN:
        return True
    return False


def main() -> None:
    settings = get_settings()
    args = parse_args()

    base_dir = Path(__file__).parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else base_dir / "data" / "raw"
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    processed_dir = out_dir / "processed" if out_dir else base_dir / "data" / "processed"
    exclude_tokens = [
        token.strip().lower()
        for token in args.exclude_jsonl_tokens.split(",")
        if token.strip()
    ]

    if jsonl_path and not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl path not found: {jsonl_path.resolve()}")
    if (not jsonl_path or args.include_raw_txt) and not raw_dir.exists():
        raise FileNotFoundError(f"raw path not found: {raw_dir.resolve()}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    chunks_path = processed_dir / "chunks.jsonl"
    if chunks_path.exists():
        chunks_path.unlink()

    chunk_id = 0
    if jsonl_path:
        total_lines = 0
        json_errors = 0
        loaded_docs = 0
        excluded_docs = 0
        skipped_chunks = 0
        total_chunks = 0
        print(f"Loading JSONL: {jsonl_path.resolve()}")
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    json_errors += 1
                    continue
                if not isinstance(obj, dict):
                    excluded_docs += 1
                    continue
                content = obj.get("text")
                if not isinstance(content, str) or not content.strip():
                    excluded_docs += 1
                    continue
                doc_meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                doc_id = obj.get("id") or f"doc_{line_no}"
                source = doc_meta.get("source") or jsonl_path.name
                title = doc_meta.get("title") or obj.get("title") or doc_id
                if _should_exclude_jsonl(
                    doc_meta,
                    content,
                    doc_id,
                    source,
                    title,
                    exclude_tokens,
                ):
                    excluded_docs += 1
                    continue
                loaded_docs += 1

                sections = doc_meta.get("sections")
                section_label = " / ".join(sections) if isinstance(sections, list) else None
                category = doc_meta.get("category")
                doc_type = doc_meta.get("doc_type") or "aihub_71762"
                tags = doc_meta.get("tags")
                if not isinstance(tags, list):
                    tags = [category] if isinstance(category, str) and category else [doc_type]

                chunks = chunk_text(
                    content,
                    max_len=args.chunk_size,
                    overlap=args.chunk_overlap,
                    add_section_prefix=False,
                    detect_headers=False,
                )
                for i, (chunk, section) in enumerate(chunks):
                    if len(chunk) < MIN_JSONL_CHUNK_LEN:
                        skipped_chunks += 1
                        continue
                    total_chunks += 1
                    chunk_section = section or section_label
                    with chunks_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "chunk_id": chunk_id,
                            "source": source,
                            "doc_type": doc_type,
                            "chunk": i,
                            "title": title,
                            "section": chunk_section,
                            "text": chunk,
                        }, ensure_ascii=False) + "\n")

                    texts.append(chunk)
                    entry = dict(doc_meta)
                    entry.update({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source": source,
                        "doc_type": doc_type,
                        "chunk": i,
                        "title": title,
                        "section": chunk_section,
                        "text": chunk,
                        "snippet": chunk[:200].replace("\n", " "),
                        "source_file": doc_meta.get("source_file") or jsonl_path.name,
                        "category": category or doc_type,
                        "disease": doc_meta.get("disease"),
                        "tags": tags,
                        "department": doc_meta.get("department"),
                        "effective_date": doc_meta.get("effective_date"),
                    })
                    meta.append(entry)
                    chunk_id += 1
        print(
            "JSONL summary:",
            f"lines={total_lines}",
            f"loaded_docs={loaded_docs}",
            f"excluded_docs={excluded_docs}",
            f"json_errors={json_errors}",
            f"chunks={total_chunks}",
            f"skipped_chunks={skipped_chunks}",
        )
    if not jsonl_path or args.include_raw_txt:
        print(f"Loading TXT: {raw_dir.resolve()}")
        for txt_file in raw_dir.glob("*.txt"):
            if txt_file.name in EXCLUDE_FROM_RAG:
                print(f"RAG excluded: {txt_file.name}")
                continue

            content = txt_file.read_text(encoding="utf-8")
            content = _strip_frontmatter(content)
            chunks = chunk_text(content, max_len=args.chunk_size, overlap=args.chunk_overlap)

            doc_type = DOC_TYPE_MAP.get(txt_file.name, "unknown")
            title = txt_file.stem
            tags = [doc_type] if doc_type else []

            for i, (chunk, section) in enumerate(chunks):
                with chunks_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "chunk_id": chunk_id,
                        "source": txt_file.name,
                        "doc_type": doc_type,
                        "chunk": i,
                        "title": title,
                        "section": section,
                        "text": chunk,
                    }, ensure_ascii=False) + "\n")

                texts.append(chunk)
                meta.append({
                    "chunk_id": chunk_id,
                    "source": txt_file.name,
                    "doc_type": doc_type,
                    "chunk": i,
                    "title": title,
                    "section": section,
                    "text": chunk,
                    "snippet": chunk[:200].replace("\n", " "),
                    "source_file": txt_file.name,
                    "category": doc_type,
                    "tags": tags,
                    "department": None,
                    "effective_date": None,
                })
                chunk_id += 1

    if not texts:
        raise ValueError("No documents to ingest.")

    excluded_sources = {m.get("source_file") for m in meta if m.get("source_file") in EXCLUDE_FROM_RAG}
    if excluded_sources:
        print(f"Warning: excluded sources found in metadata: {sorted(excluded_sources)}")

    print(f"Building embeddings ({len(texts)} chunks)")
    vectors = np.asarray(embed_texts(texts), dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / "faiss.index"
        metadata_path = out_dir / "metadata.json"
    else:
        index_path = Path(settings.faiss_index_path)
        metadata_path = Path(settings.metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("chunks saved:", chunks_path.resolve())
    print("FAISS index saved:", index_path.resolve())
    print("metadata saved:", metadata_path.resolve())


if __name__ == "__main__":
    main()
