from django.core.management.base import BaseCommand

from chatbot.services.cache_service import clear_cache


class Command(BaseCommand):
    help = "Clear chatbot cache entries from the database."

    def handle(self, *args, **options):
        deleted = clear_cache()
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} cache entries."))
