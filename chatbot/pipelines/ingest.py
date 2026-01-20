from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    doc_id: int
    title: str
    text: str
    snippet: str
    source_file: str
    category: str
    tags: list[str]
    department: str | None = None
    effective_date: str | None = None
    section: str | None = None


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

def load_documents(source_dir: Path) -> List[Document]:
    """Load and process documents from the source directory."""
    docs: List[Document] = []
    doc_id = 0
    
    for path in sorted(source_dir.glob("*.txt")):
        try:
            if path.name == '.gitkeep':
                continue
                
            content = path.read_text(encoding="utf-8").strip()
            front_matter: dict[str, str] = {}
            text = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    front_matter_text = parts[1].strip()
                    text = parts[2].strip()
                    for line in front_matter_text.splitlines():
                        if ":" not in line:
                            continue
                        key, value = line.split(":", 1)
                        front_matter[key.strip().lower()] = value.strip()

            title = front_matter.get("title") or path.stem
            category = front_matter.get("category") or DOC_TYPE_MAP.get(path.name, "unknown")
            tags = [t.strip() for t in front_matter.get("tags", "").split(",") if t.strip()]
            if category and category not in tags:
                tags.append(category)
            department = front_matter.get("department")
            effective_date = front_matter.get("effective_date") or front_matter.get("date")

            # Snippet은 공백을 정리한 텍스트 기준으로 생성
            normalized = " ".join(text.split())
            snippet = normalized[:200]
            if len(normalized) > 200:
                for punct in (".", "!", "?", "다."):
                    last_punct = snippet.rfind(punct)
                    if last_punct > 100:
                        snippet = snippet[: last_punct + 1]
                        break

            docs.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    snippet=snippet,
                    source_file=path.name,
                    category=category,
                    tags=tags,
                    department=department,
                    effective_date=effective_date,
                )
            )
            doc_id += 1
            
        except Exception as e:
            print(f"Error loading document {path}: {str(e)}")
            continue
            
    return docs


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


def _find_section_headers(text: str) -> List[tuple[int, str]]:
    headers: List[tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(True):
        if _is_section_header(line):
            header = line.strip().strip("#").strip("[]").strip()
            if header:
                headers.append((pos, header))
        pos += len(line)
    return headers


def _section_for_offset(headers: List[tuple[int, str]], offset: int) -> str:
    current = ""
    for pos, header in headers:
        if pos <= offset:
            current = header
        else:
            break
    return current


def split_into_chunks(docs: Iterable[Document], chunk_size: int = 500, overlap: int = 100) -> List[Document]:
    chunks: List[Document] = []
    chunk_id = 0
    for doc in docs:
        text = doc.text
        headers = _find_section_headers(text)
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()
            section = _section_for_offset(headers, start)
            if section:
                chunk_text = f"{section}\n{chunk_text}"
            snippet = chunk_text[:200].replace("\n", " ")
            chunks.append(
                Document(
                    doc_id=chunk_id,
                    title=doc.title,
                    text=chunk_text,
                    snippet=snippet,
                    source_file=doc.source_file,
                    category=doc.category,
                    tags=doc.tags,
                    department=doc.department,
                    effective_date=doc.effective_date,
                    section=section or None,
                )
            )
            chunk_id += 1
            start += chunk_size - overlap
    return chunks


def build_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def save_metadata(chunks: List[Document], path: Path) -> None:
    payload = {
        str(chunk.doc_id): {
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "snippet": chunk.snippet,
            "source_file": chunk.source_file,
            "category": chunk.category,
            "tags": chunk.tags,
            "department": chunk.department,
            "effective_date": chunk.effective_date,
            "section": chunk.section,
        }
        for chunk in chunks
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(
    raw_dir: Path,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> None:
    docs = load_documents(raw_dir)
    if not docs:
        raise RuntimeError("raw_docs 폴더에 문서가 없습니다.")

    chunks = split_into_chunks(docs, chunk_size=chunk_size, overlap=overlap)
    texts = [chunk.text for chunk in chunks]
    embeddings = build_embeddings(texts, embedding_model)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    save_index(index, index_path)
    save_metadata(chunks, metadata_path)
    print(f"FAISS 인덱스 생성 완료: {len(chunks)}개 청크")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    index_path = base_dir / "data" / "faiss.index"
    metadata_path = base_dir / "data" / "metadata.json"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    ingest(
        raw_dir=raw_dir,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )
