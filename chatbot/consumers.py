import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from .models import ChatMessage

User = get_user_model()

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = 'chat_room'
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        user = self.scope['user']
        
        # Save message to database
        response = await self.save_message(user, message)
        
        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'response': response,
                'username': user.username if user.is_authenticated else 'Anonymous',
            }
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event['message']
        response = event['response']
        username = event['username']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message,
            'response': response,
            'username': username,
        }))

    @database_sync_to_async
    def save_message(self, user, message):
        # This is a simple echo response. Replace with your actual chatbot logic.
        response = f"You said: {message}"
        
        if user.is_authenticated:
            ChatMessage.objects.create(
                user=user,
                message=message,
                response=response
            )
        return response
