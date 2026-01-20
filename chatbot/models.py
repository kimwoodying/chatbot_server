import os

from django.db import models


# Department model - 진료과 정보
class Department(models.Model):
    name = models.CharField("진료과명", max_length=100, unique=True)
    code = models.CharField("진료과 코드", max_length=20, unique=True)
    description = models.TextField("설명", blank=True)
    created_at = models.DateTimeField("생성 시각", auto_now_add=True)
    
    class Meta:
        ordering = ["name"]
        verbose_name = "진료과"
        verbose_name_plural = "진료과"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.code})"


# Doctor model - 의사 정보
class Doctor(models.Model):
    name = models.CharField("의사명", max_length=100)
    department = models.ForeignKey(
        Department, 
        on_delete=models.CASCADE, 
        related_name="doctors",
        verbose_name="진료과"
    )
    title = models.CharField("직책", max_length=50, default="전문의")
    specialty = models.CharField("전문 분야", max_length=200, blank=True)
    is_active = models.BooleanField("활성화", default=True)
    created_at = models.DateTimeField("생성 시각", auto_now_add=True)
    
    class Meta:
        ordering = ["department", "name"]
        verbose_name = "의사"
        verbose_name_plural = "의사"
        unique_together = [["name", "department"]]
    
    def __str__(self) -> str:
        return f"{self.name} {self.title} ({self.department.name})"


class ChatMessage(models.Model):
    session_id = models.CharField(
        "세션 ID",
        max_length=255,
        blank=True,
        help_text="사용자 구분용 세션 식별자",
    )
    user_question = models.TextField("사용자 질문")
    bot_answer = models.TextField("챗봇 응답")
    sources = models.JSONField(
        "참고 자료",
        blank=True,
        null=True,
        help_text="RAG 검색 결과(JSON)"
    )
    metadata = models.JSONField(
        "요청 메타데이터",
        blank=True,
        null=True,
        help_text="사용자 장치 또는 기타 부가 정보"
    )
    created_at = models.DateTimeField("생성 시각", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "채팅 내역"
        verbose_name_plural = "채팅 내역"

    def __str__(self) -> str:
        return f"{self.session_id or 'anonymous'}: {self.user_question[:30]}"


class ChatCache(models.Model):
    """
    질문 + 컨텍스트 조합에 대한 LLM 응답 캐시.
    같은 질문/컨텍스트 조합이면 DB에서 바로 꺼내서 LLM 호출을 생략.
    """

    cache_key = models.TextField(blank=True)                  # 캐시 키(디버깅용)
    query_hash = models.CharField(max_length=64, unique=True)  # SHA-256 해시
    intent = models.CharField(max_length=32, blank=True)      # tool/rag/static 등
    cache_scope = models.CharField(max_length=32, blank=True) # query_only/rag_context 등
    normalized_query = models.TextField(blank=True)           # 정규화된 질문
    rag_index_version = models.CharField(max_length=32, blank=True)
    top_k = models.PositiveIntegerField(default=0)
    prompt_version = models.CharField(max_length=32, blank=True)
    query = models.TextField()                                # 원본 질문
    context = models.TextField(blank=True)                    # 사용된 컨텍스트 텍스트
    context_hash = models.CharField(max_length=64, blank=True)
    sources_hash = models.CharField(max_length=64, blank=True)
    response = models.TextField()                             # LLM 생성 응답
    sources = models.JSONField(blank=True, null=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    hit_count = models.PositiveIntegerField(default=1)        # 몇 번이나 재사용됐는지
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chatbot_cache"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.query[:30]}... ({self.hit_count} hits)"


# Tool data model: used by reservation tools for create/lookup/cancel/history.
class Reservation(models.Model):
    STATUS_CHOICES = [
        ("pending", "pending"),
        ("confirmed", "confirmed"),
        ("cancelled", "cancelled"),
    ]

    session_id = models.CharField(max_length=255, blank=True)
    patient_name = models.CharField(max_length=100, blank=True)
    patient_phone = models.CharField(max_length=50, blank=True)
    department = models.CharField(max_length=100)
    reason = models.TextField(blank=True)
    requested_time_text = models.CharField(max_length=100, blank=True)
    channel = models.CharField(max_length=50, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    cancel_reason = models.TextField(blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "예약"
        verbose_name_plural = "예약"

    def __str__(self) -> str:
        return f"{self.department} ({self.status})"


# External reservation table in the hospital DB (read-only mapping).
HOSPITAL_RESERVATION_TABLE = os.getenv("HOSPITAL_RESERVATION_TABLE", "patients_appointment")


class HospitalReservation(models.Model):
    id = models.CharField(primary_key=True, max_length=64)
    title = models.CharField(max_length=200, blank=True)
    patient_identifier = models.CharField(max_length=100, db_column="patient_identifier")
    patient_name = models.CharField(max_length=100, blank=True)
    doctor_name = models.CharField(max_length=100, blank=True)
    doctor_department = models.CharField(max_length=100, blank=True)
    start_time = models.DateTimeField(db_column="start_time")
    end_time = models.DateTimeField(db_column="end_time", null=True, blank=True)
    status = models.CharField(max_length=50)
    memo = models.TextField(db_column="memo", blank=True, null=True)

    class Meta:
        managed = False
        db_table = HOSPITAL_RESERVATION_TABLE


# Tool data model: used by wait_status tool for department queue info.
class WaitStatus(models.Model):
    department = models.CharField(max_length=100)
    current_waiting = models.PositiveIntegerField(default=0)
    estimated_minutes = models.PositiveIntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    updated_by = models.CharField(max_length=100, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-last_updated"]
        verbose_name = "대기 상태"
        verbose_name_plural = "대기 상태"

    def __str__(self) -> str:
        return f"{self.department} ({self.current_waiting})"


# Tool data model: used by notification tool for outbound requests.
class Notification(models.Model):
    STATUS_CHOICES = [
        ("pending", "pending"),
        ("sent", "sent"),
        ("failed", "failed"),
    ]

    session_id = models.CharField(max_length=255, blank=True)
    channel = models.CharField(max_length=50)
    target = models.CharField(max_length=100, blank=True)
    message = models.TextField()
    schedule_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    sent_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "알림"
        verbose_name_plural = "알림"

    def __str__(self) -> str:
        return f"{self.channel} ({self.status})"


class ToolAuditLog(models.Model):
    request_id = models.CharField(max_length=64, blank=True)
    session_id = models.CharField(max_length=255, blank=True)
    user_id = models.CharField(max_length=255, blank=True)
    tool_name = models.CharField(max_length=64)
    status = models.CharField(max_length=32)
    error_code = models.CharField(max_length=64, blank=True)
    latency_ms = models.PositiveIntegerField(default=0)
    metadata = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "툴 감사로그"
        verbose_name_plural = "툴 감사로그"

    def __str__(self) -> str:
        return f"{self.tool_name} ({self.status})"
