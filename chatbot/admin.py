from django.contrib import admin
from .models import (
    ChatMessage,
    ChatCache,
    Reservation,
    WaitStatus,
    Notification,
    ToolAuditLog,
    HospitalReservation,
)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session_id", "user_question", "bot_answer", "created_at")
    list_filter = ("created_at",)
    search_fields = ("session_id", "user_question", "bot_answer")


@admin.register(ChatCache)
class ChatCacheAdmin(admin.ModelAdmin):
    list_display = ("query", "intent", "hit_count", "expires_at", "created_at")
    list_filter = ("intent", "cache_scope", "created_at")
    search_fields = ("query", "response")
    readonly_fields = ("query_hash", "created_at")


@admin.register(Reservation)
class ReservationAdmin(admin.ModelAdmin):
    list_display = ("department", "patient_name", "status", "requested_time_text", "created_at")
    list_filter = ("status", "department", "created_at")
    search_fields = ("patient_name", "patient_phone", "department")


@admin.register(WaitStatus)
class WaitStatusAdmin(admin.ModelAdmin):
    list_display = ("department", "current_waiting", "estimated_minutes", "last_updated", "updated_by")
    list_filter = ("department", "last_updated")


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ("channel", "target", "status", "created_at", "sent_at")
    list_filter = ("status", "channel", "created_at")
    search_fields = ("target", "message")


@admin.register(ToolAuditLog)
class ToolAuditLogAdmin(admin.ModelAdmin):
    list_display = ("tool_name", "status", "error_code", "latency_ms", "created_at")
    list_filter = ("tool_name", "status", "created_at")
    search_fields = ("request_id", "session_id", "error_code")


@admin.register(HospitalReservation)
class HospitalReservationAdmin(admin.ModelAdmin):
    list_display = ("patient_name", "doctor_name", "doctor_department", "start_time", "status", "memo")
    list_filter = ("status", "doctor_department", "start_time")
    search_fields = ("patient_name", "patient_identifier", "doctor_name")
    
    # 외부 테이블이므로 수정 방지 (선택 사항)
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False

