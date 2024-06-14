from rest_framework import serializers
from .models import Chat
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username']

class ChatSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)  # Use nested serialization for user info

    class Meta:
        model = Chat
        fields = ['id', 'user', 'message', 'response', 'created_at', 'visible']