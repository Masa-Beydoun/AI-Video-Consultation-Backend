# consulting/serializers/consultant_application_serializer.py
from rest_framework import serializers
from consulting.models.consultant_application import ConsultantApplication
from consulting.models.resource import Resource
from django.contrib.contenttypes.models import ContentType
from consulting.models.domain import Domain
from consulting.models.subdomain import SubDomain

class ConsultantApplicationSerializer(serializers.ModelSerializer):
    # Accept file upload for photo
    photo_file = serializers.FileField(write_only=True, required=False)
    photo = serializers.SerializerMethodField(read_only=True)
    domain_name = serializers.CharField(write_only=True, required=False)
    sub_domain_name = serializers.CharField(write_only=True, required=False)
    class Meta:
        model = ConsultantApplication
        fields = [
            "id", "photo", "photo_file", "location", "description", "cost",
            "years_experience", "languages", "status", "admin_notes",
            "reviewed_at", "created_at", "user", "reviewed_by",
            "domain_name", "sub_domain_name",
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
        domain_name = validated_data.pop("domain_name", None)
        sub_domain_name = validated_data.pop("sub_domain_name", None)

        # Domain
        if domain_name:
            domain = Domain.objects.filter(name=domain_name, status="approved").first()
            if not domain:
                domain = Domain.objects.create(name=domain_name, status="pending")
            validated_data["domain"] = domain

        # SubDomain
        if sub_domain_name and validated_data.get("domain"):
            sub_domain = SubDomain.objects.filter(
                name=sub_domain_name,
                domain=validated_data["domain"],
                status="approved"
            ).first()
            if not sub_domain:
                sub_domain = SubDomain.objects.create(
                    name=sub_domain_name,
                    domain=validated_data["domain"],
                    status="pending"
                )
            validated_data["sub_domain"] = sub_domain

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

        # Domain / SubDomain logic
        domain_name = validated_data.pop("domain_name", None)
        sub_domain_name = validated_data.pop("sub_domain_name", None)

        if domain_name:
            domain = Domain.objects.filter(name=domain_name, status="approved").first()
            if not domain:
                domain = Domain.objects.create(name=domain_name, status="pending")
            validated_data["domain"] = domain

        if sub_domain_name and validated_data.get("domain"):
            sub_domain = SubDomain.objects.filter(
                name=sub_domain_name,
                domain=validated_data["domain"],
                status="approved"
            ).first()
            if not sub_domain:
                sub_domain = SubDomain.objects.create(
                    name=sub_domain_name,
                    domain=validated_data["domain"],
                    status="pending"
                )
            validated_data["sub_domain"] = sub_domain

        # Update the instance
        instance = super().update(instance, validated_data)

        # Handle photo
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
