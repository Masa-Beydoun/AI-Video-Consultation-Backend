from rest_framework import serializers
from .models import Message
from .models import Chat
from consulting.models.consultant import Consultant

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'chat', 'sender', 'text', 'sent_at']
        read_only_fields = ['id', 'sent_at']

class ConsultantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Consultant
        fields = ['id']


class ChatSerializer(serializers.ModelSerializer):
    consultant = ConsultantSerializer()

    class Meta:
        model = Chat
        fields = ['id', 'title', 'consultant', 'created_at', 'modified_at']