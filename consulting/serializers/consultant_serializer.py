from rest_framework import serializers
from consulting.models.consultant import Consultant

class ConsultantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Consultant
        fields = '__all__'
        read_only_fields = ['validated', 'validated_by', 'validated_at', 'added_at', 'rating', 'review_count']
