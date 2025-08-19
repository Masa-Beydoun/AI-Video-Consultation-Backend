from .resource_serializer import ResourceSerializer
from rest_framework import serializers
from consulting.models.consultant_application import ConsultantApplication
from consulting.models.resource import Resource
from django.contrib.contenttypes.models import ContentType


class ConsultantApplicationSerializer(serializers.ModelSerializer):
    # Nested resource serializer for reading
    resources = ResourceSerializer(many=True, read_only=True, source='resource_set')

    # For writing (uploading files)
    uploaded_files = serializers.ListField(
        child=serializers.FileField(),
        write_only=True,
        required=False
    )

    class Meta:
        model = ConsultantApplication
        fields = '__all__'
        read_only_fields = ['user', 'status', 'submitted_at']
    def create(self, validated_data):
        uploaded_files = validated_data.pop('uploaded_files', [])
        application = ConsultantApplication.objects.create(**validated_data)

        # Save uploaded files as resources
        content_type = ContentType.objects.get_for_model(ConsultantApplication)
        for file in uploaded_files:
            Resource.objects.create(
                file_path=file,
                relation_type=content_type,
                relation_id=application.id
            )

        return application

    def update(self, instance, validated_data):
        uploaded_files = validated_data.pop('uploaded_files', [])
        instance = super().update(instance, validated_data)

        # Optionally allow adding new files on update
        if uploaded_files:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            for file in uploaded_files:
                Resource.objects.create(
                    file_path=file,
                    relation_type=content_type,
                    relation_id=instance.id
                )

        return instance