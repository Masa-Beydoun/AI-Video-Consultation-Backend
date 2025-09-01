from rest_framework import serializers
from .models import Message
from .models import Chat
from consulting.models.consultant import Consultant, User

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'chat', 'sender', 'text', 'sent_at']
        read_only_fields = ['id', 'sent_at']

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'phone_number', 'role', 'gender']


class ConsultantSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Consultant
        fields = '__all__'


class ChatSerializer(serializers.ModelSerializer):
    consultant = ConsultantSerializer()

    class Meta:
        model = Chat
        fields = ['id', 'title', 'consultant', 'created_at', 'modified_at']