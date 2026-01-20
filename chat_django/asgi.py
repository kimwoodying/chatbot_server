import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

from chatbot import routing as chatbot_routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chat_django.settings")

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(chatbot_routing.websocket_urlpatterns)
        ),
    }
)
