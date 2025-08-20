from rest_framework import serializers
from consulting.models.consultant import Consultant
from django.contrib.auth import get_user_model

User = get_user_model()

class ConsultantSerializer(serializers.ModelSerializer):
    # Show the username of the user who validated, read-only
    validated_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Consultant
        # Explicit fields for clarity
        fields = [
            'id', 'name', 'email', 'phone', 'specialty',
            'validated', 'validated_by', 'validated_at',
            'added_at', 'rating', 'review_count'
        ]
        read_only_fields = ['validated', 'validated_by', 'validated_at', 'added_at', 'rating', 'review_count']

    def create(self, validated_data):
        # Ensure a new consultant is not validated by default
        validated_data['validated'] = False
        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Prevent updating read-only fields manually
        for field in self.Meta.read_only_fields:
            validated_data.pop(field, None)
        return super().update(instance, validated_data)
