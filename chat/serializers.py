from rest_framework import serializers
from .models import Message
from .models import Chat
from .models import MessageResource
from consulting.models.consultant import Consultant
from consulting.models.user import User
from consulting.models.domain import Domain
from consulting.models.subdomain import SubDomain
from consulting.models.resource import Resource

class UserMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'chat', 'sender', 'text', 'sent_at']
        read_only_fields = ['id', 'sent_at']

class ResourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resource
        fields =  '__all__'

class ConsultantMessageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Message
        fields =  '__all__'

class MessageResourceSerializer(serializers.ModelSerializer):

    resource = ResourceSerializer(read_only=True)

    class Meta:
        model = MessageResource
        fields =  '__all__'

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'phone_number', 'role', 'gender']

class DomainSerializer(serializers.ModelSerializer):
    class Meta:
        model = Domain
        fields = '__all__'

class SubDomainSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubDomain
        fields = '__all__'


class ConsultantSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    domain = DomainSerializer(read_only=True)
    sub_domain = SubDomainSerializer(read_only=True)

    class Meta:
        model = Consultant
        fields = '__all__'


class ChatSerializer(serializers.ModelSerializer):
    consultant = ConsultantSerializer()

    class Meta:
        model = Chat
        fields = ['id', 'title', 'consultant', 'created_at', 'modified_at']