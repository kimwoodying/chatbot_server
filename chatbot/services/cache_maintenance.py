from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone
from filelock import FileLock, Timeout

from chatbot.services.cache_service import clear_cache

logger = logging.getLogger(__name__)

_SCHEDULER_STARTED = False


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _state_path() -> Path:
    raw = os.getenv("CACHE_CLEAR_STATE_PATH")
    if raw:
        return Path(raw)
    return Path(settings.BASE_DIR) / "cache_clear_state.json"


def _lock_path() -> Path:
    raw = os.getenv("CACHE_CLEAR_LOCK_PATH")
    if raw:
        return Path(raw)
    return Path(settings.BASE_DIR) / "cache_clear.lock"


def _read_last_run_date() -> str | None:
    path = _state_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    last_run = payload.get("last_run_date")
    if isinstance(last_run, str) and last_run:
        return last_run
    return None


def _write_last_run_date(date_str: str) -> None:
    path = _state_path()
    payload = {"last_run_date": date_str}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _seconds_until_next_run(now, hour: int, minute: int) -> float:
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return max((next_run - now).total_seconds(), 60.0)


def _should_run(now, last_run_date: str | None, hour: int, minute: int) -> bool:
    today = now.date().isoformat()
    if last_run_date == today:
        return False
    scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return now >= scheduled


def _run_clear() -> None:
    lock = FileLock(str(_lock_path()))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return
    try:
        close_old_connections()
        deleted = clear_cache()
        logger.info("cache clear: deleted=%s", deleted)
        _write_last_run_date(timezone.localdate().isoformat())
    finally:
        close_old_connections()
        try:
            lock.release()
        except Exception:
            pass


def _run_loop() -> None:
    hour = max(0, min(23, _get_int("CACHE_CLEAR_HOUR", 4)))
    minute = max(0, min(59, _get_int("CACHE_CLEAR_MINUTE", 0)))
    retry_delay = max(60, _get_int("CACHE_CLEAR_RETRY_SECONDS", 300))

    while True:
        now = timezone.localtime()
        last_run_date = _read_last_run_date()

        if _should_run(now, last_run_date, hour, minute):
            _run_clear()
            last_run_date = _read_last_run_date()

        if last_run_date == now.date().isoformat():
            sleep_seconds = _seconds_until_next_run(now, hour, minute)
        elif now < now.replace(hour=hour, minute=minute, second=0, microsecond=0):
            sleep_seconds = _seconds_until_next_run(now, hour, minute)
        else:
            sleep_seconds = float(retry_delay)
        time.sleep(sleep_seconds)


def _should_skip_for_command() -> bool:
    argv = " ".join(os.sys.argv)
    for cmd in ("migrate", "makemigrations", "collectstatic", "shell", "check", "test"):
        if cmd in argv:
            return True
    if "runserver" in argv and os.environ.get("RUN_MAIN") != "true":
        return True
    return False


def start_cache_clear_scheduler() -> None:
    global _SCHEDULER_STARTED
    if _SCHEDULER_STARTED:
        return
    if not _get_bool("CACHE_CLEAR_ENABLED", True):
        return
    if _should_skip_for_command():
        return
    _SCHEDULER_STARTED = True
    thread = threading.Thread(target=_run_loop, name="cache-clear-scheduler", daemon=True)
    thread.start()
