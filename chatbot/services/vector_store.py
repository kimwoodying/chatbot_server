# chatbot/services/vector_store.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from chatbot.config import get_settings

settings = get_settings()


class VectorStore:
    def __init__(self, index_path: Path, metadata_path: Path):
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS 인덱스 또는 메타데이터 파일이 존재하지 않습니다. 먼저 ingest 스크립트를 실행하세요."
            )

        # ensure string path for faiss, use as_posix() to minimize encoding issues on some windows setups
        self._index = faiss.read_index(index_path.as_posix())

        with metadata_path.open(encoding="utf-8") as f:
            raw_meta = json.load(f)

        def _get_key(item: dict) -> int:
            # ingest 산출물이 id/chunk_id/doc_id 중 무엇을 쓰든 지원
            for k in ("id", "chunk_id", "doc_id"):
                if k in item:
                    return int(item[k])
            raise KeyError("metadata item에 id/chunk_id/doc_id 중 어떤 키도 없습니다.")

        if isinstance(raw_meta, list):
            self._metadata = {_get_key(item): item for item in raw_meta}
        elif isinstance(raw_meta, dict):
            # dict 형식(예: {"0": {...}, "1": {...}})도 지원
            self._metadata = {int(k): v for k, v in raw_meta.items()}
        else:
            raise TypeError(f"지원하지 않는 metadata 형식입니다: {type(raw_meta)}")

    def search(self, query_vector, top_k: int = 5):
        """
        IndexFlatL2 기반:
        - distances: 작을수록 더 유사함
        반환은 (distance, meta) 튜플 리스트
        """
        q = np.array(query_vector, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        distances, indices = self._index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            meta = self._metadata.get(int(idx))
            if not meta:
                continue

            meta_with_id = {"id": int(idx), **meta}
            results.append((float(dist), meta_with_id))

        return results


@lru_cache(maxsize=1)
def get_vector_store():
    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.metadata_path)
    return VectorStore(index_path, metadata_path)
