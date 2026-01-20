from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from chatbot.config import get_settings
from chatbot.pipelines.ingest import ingest


class Command(BaseCommand):
    help = "문서를 임베딩하여 FAISS 인덱스를 생성합니다."

    def add_arguments(self, parser):
        parser.add_argument(
            "--raw-dir",
            type=str,
            default=None,
            help="원본 문서(.txt)가 위치한 디렉터리 경로",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=500,
            help="청크 길이(문자 기준)",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=100,
            help="청크 간 겹침(문자)",
        )

    def handle(self, *args, **options):
        settings = get_settings()
        raw_dir = options["raw_dir"]
        if raw_dir is None:
            raw_dir_path = settings.data_dir / "raw"
        else:
            raw_dir_path = Path(raw_dir)

        raw_dir_path.mkdir(parents=True, exist_ok=True)
        settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

        self.stdout.write(self.style.NOTICE("문서 ingest를 시작합니다."))
        try:
            ingest(
                raw_dir=raw_dir_path,
                index_path=settings.faiss_index_path,
                metadata_path=settings.metadata_path,
                embedding_model=settings.embedding_model,
                chunk_size=options["chunk_size"],
                overlap=options["overlap"],
            )
        except RuntimeError as exc:
            raise CommandError(str(exc))
        except Exception as exc:  # pragma: no cover
            raise CommandError(f"Ingest 실패: {exc}") from exc
        else:
            self.stdout.write(self.style.SUCCESS("FAISS 인덱스 생성이 완료되었습니다."))
