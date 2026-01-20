from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatbot"

    def ready(self):
        from chatbot.services.cache_maintenance import start_cache_clear_scheduler

        start_cache_clear_scheduler()
