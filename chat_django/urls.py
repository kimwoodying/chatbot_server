from django.contrib import admin
from django.urls import path, include

from chatbot import views as chatbot_views

urlpatterns = [
    path("", chatbot_views.reservation_page, name="reservation_page"),
    path("api/reservation/options/", chatbot_views.reservation_options, name="reservation_options"),
    path("admin/", admin.site.urls),
    path("api/chat/", include("chatbot.urls")),
]
