# consulting/serializers/consultant_application_serializer.py
from rest_framework import serializers
from consulting.models.consultant_application import ConsultantApplication
from consulting.models.resource import Resource
from django.contrib.contenttypes.models import ContentType

class ConsultantApplicationSerializer(serializers.ModelSerializer):
    # Accept file upload for photo
    photo_file = serializers.FileField(write_only=True, required=False)
    photo = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = ConsultantApplication
        fields = [
            "id", "photo", "photo_file", "location", "description", "cost",
            "years_experience", "languages", "status", "admin_notes",
            "reviewed_at", "created_at", "user", "domain", "sub_domain", "reviewed_by"
        ]
        read_only_fields = ["user", "status", "created_at", "reviewed_by", "reviewed_at"]

    def get_photo(self, obj):
        """Return Resource details if photo exists"""
        if obj.photo:
            return {
                "id": obj.photo.id,
                "url": obj.photo.file_path.url if obj.photo.file_path else None
            }
        return None

    def create(self, validated_data):
        photo_file = validated_data.pop("photo_file", None)

        application = ConsultantApplication.objects.create(**validated_data)

        if photo_file:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            resource = Resource.objects.create(
                file_path=photo_file,
                relation_type=content_type,
                relation_id=application.id,
            )
            application.photo = resource
            application.save(update_fields=["photo"])

        return application

    def update(self, instance, validated_data):
        photo_file = validated_data.pop("photo_file", None)

        instance = super().update(instance, validated_data)

        if photo_file:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            resource = Resource.objects.create(
                file_path=photo_file,
                relation_type=content_type,
                relation_id=instance.id,
            )
            instance.photo = resource
            instance.save(update_fields=["photo"])

        return instance
