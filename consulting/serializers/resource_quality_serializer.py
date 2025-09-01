# consulting/serializers/resource_quality_serializer.py
from rest_framework import serializers
from consulting.models import ResourceQualityCheck

class ResourceQualityCheckSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResourceQualityCheck
        fields = ['id', 'resource', 'status', 'quality_report', 'checked_at']
        read_only_fields = ['status', 'quality_report', 'checked_at']
