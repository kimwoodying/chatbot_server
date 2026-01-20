import json
import logging
import uuid
from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from chatbot.models import ChatMessage, Department, Doctor
from chatbot.services.rag import run_rag_with_cache

logger = logging.getLogger(__name__)

AUTH_REQUIRED_REPLY = "로그인 후 이용해 주세요, 전화 문의는 대표번호 1577-3330으로 부탁드립니다."
RESERVATION_LOGIN_GUARD_CUES = [
    "예약",
    "예약내역",
    "예약 내역",
    "예약이력",
    "예약 이력",
    "예약 기록",
    "예약조회",
    "예약 조회",
    "예약확인",
    "예약 확인",
    "예약시간",
    "예약 시간",
    "예약일정",
    "예약 일정",
    "예약스케줄",
    "예약 스케줄",
    "예약취소",
    "예약 취소",
    "예약변경",
    "예약 변경",
]


def _load_department_fallback() -> list[dict]:
    path = Path(__file__).resolve().parent / "data" / "raw" / "departments.txt"
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    departments = []
    collecting = False
    for line in lines:
        if not line:
            continue
        if "운영 진료과" in line:
            collecting = True
            continue
        if collecting and ("진료과 설명" in line or line.startswith("[")):
            break
        if collecting and line.endswith("과") and len(line) <= 12:
            departments.append({"name": line, "code": "", "description": ""})
    return departments


@require_GET
def reservation_options(request):
    source = "db"
    departments = []
    doctors = []
    try:
        departments = [
            {
                "name": dept.name,
                "code": dept.code,
                "description": dept.description or "",
            }
            for dept in Department.objects.all().order_by("name")
        ]
    except Exception as exc:
        logger.warning("reservation options: department load failed: %s", exc)
        departments = []
    if not departments:
        departments = _load_department_fallback()
        source = "fallback"
    try:
        doctors = [
            {
                "name": doctor.name,
                "department": doctor.department.name,
                "title": doctor.title,
                "specialty": doctor.specialty or "",
            }
            for doctor in Doctor.objects.filter(is_active=True).select_related("department")
        ]
    except Exception as exc:
        logger.warning("reservation options: doctor load failed: %s", exc)
        doctors = []
    return JsonResponse({"departments": departments, "doctors": doctors, "source": source})


def reservation_page(request):
    return render(request, "chatbot/reservation.html")


@csrf_exempt
@require_POST
# API endpoint: handles chat POST requests and returns chatbot response JSON.
def chat_view(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)

    message = payload.get("message")
    if not message:
        return JsonResponse({"error": "message 필드가 필요합니다."}, status=400)

    session_id = payload.get("session_id", "")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    auth_keys = {
        "patient_id",
        "patient_identifier",
        "patient_phone",
        "account_id",
        "patient_pk",
        "auth_user_id",
        "user_id",
    }
    blocked_auth_keys = auth_keys | {"patientId", "verified_user"}
    # Drop client-supplied auth metadata to prevent spoofing.
    for key in blocked_auth_keys:
        metadata.pop(key, None)
    has_auth = any(metadata.get(key) for key in auth_keys)
    if session_id:
        # 최근 10개 메시지에서 컨텍스트 복원 (Slot-Filling Recovery)
        recent_messages = ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:10]
        context_keys = [
            "patient_id", "patient_identifier", "patient_phone",
            "account_id", "patient_pk", "auth_user_id", "user_id",
            "doctor_name", "doctor_id", "doctor", "doctorId", "doctor_code",
            "department", "dept"
        ]
        
        for msg in recent_messages:
            if not isinstance(msg.metadata, dict):
                continue
            for key in context_keys:
                if key not in metadata and msg.metadata.get(key):
                    # 인증된 요청이 아니면 민감 정보 복원 방지
                    if key in auth_keys and not has_auth:
                        continue
                    metadata[key] = msg.metadata.get(key)
    request_id = payload.get("request_id") or metadata.get("request_id") or uuid.uuid4().hex
    metadata["request_id"] = request_id

    normalized_message = message.strip()
    guard_match = (
        not has_auth
        and normalized_message
        and any(cue in normalized_message for cue in RESERVATION_LOGIN_GUARD_CUES)
    )
    result = None
    if guard_match:
        logger.info(
            "chat auth gate: request_id=%s session_id=%s",
            request_id,
            session_id,
        )
        result = {"reply": AUTH_REQUIRED_REPLY, "sources": []}

    if result is None:
        try:
            logger.info(
                "chat request: request_id=%s session_id=%s message_len=%s",
                request_id,
                session_id,
                len(message),
            )
            result = run_rag_with_cache(message, session_id=session_id, metadata=metadata)
        except FileNotFoundError as exc:
            return JsonResponse(
                {
                    "error": "지식 베이스가 준비되지 않았습니다. 먼저 문서를 색인화해주세요.",
                    "detail": str(exc),
                },
                status=503,
            )
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover
            return JsonResponse({"error": f"RAG 파이프라인 오류: {exc}"}, status=500)

    hidden_sources = []
    ChatMessage.objects.create(
        session_id=session_id,
        user_question=message,
        bot_answer=result.get("reply", ""),
        sources=hidden_sources,
        metadata=metadata,
    )

    result_with_id = dict(result)
    result_with_id["request_id"] = request_id
    result_with_id["sources"] = hidden_sources
    return JsonResponse(result_with_id, status=200)
