# consulting/serializers/consultant_application_serializer.py
from rest_framework import serializers
from consulting.models.consultant_application import ConsultantApplication
from consulting.models.resource import Resource
from consulting.models.domain import Domain
from consulting.models.subdomain import SubDomain
from django.contrib.contenttypes.models import ContentType

class ConsultantApplicationSerializer(serializers.ModelSerializer):
    # Accept file upload for photo
    photo_file = serializers.FileField(write_only=True, required=False)
    photo = serializers.SerializerMethodField(read_only=True)

    # Accept multiple files
    files = serializers.ListField(
        child=serializers.FileField(),
        write_only=True,
        required=False
    )
    uploaded_files = serializers.SerializerMethodField(read_only=True)

    domain_name = serializers.CharField(write_only=True, required=False)
    sub_domain_name = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = ConsultantApplication
        fields = [
            "id", "photo", "photo_file", "location", "description", "cost",
            "years_experience", "languages", "status", "admin_notes",
            "reviewed_at", "created_at", "user", "reviewed_by",
            "domain_name", "sub_domain_name",
            "files", "uploaded_files"
        ]
        read_only_fields = ["user", "status", "created_at", "reviewed_by", "reviewed_at"]

    def get_photo(self, obj):
        if obj.photo:
            return {
                "id": obj.photo.id,
                "url": obj.photo.file_path.url if obj.photo.file_path else None
            }
        return None

    def get_uploaded_files(self, obj):
        resources = Resource.objects.filter(
            relation_type__model='consultantapplication',
            relation_id=obj.id
        )
        return [
            {"id": r.id, "url": r.file_path.url if r.file_path else None} for r in resources
        ]

    def create(self, validated_data):
        files = validated_data.pop("files", [])
        photo_file = validated_data.pop("photo_file", None)
        domain_name = validated_data.pop("domain_name", None)
        sub_domain_name = validated_data.pop("sub_domain_name", None)

        # Domain logic
        if domain_name:
            domain = Domain.objects.filter(name=domain_name, status="approved").first()
            if not domain:
                domain = Domain.objects.create(name=domain_name, status="pending")
            validated_data["domain"] = domain

        # SubDomain logic
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

        # Create application
        application = ConsultantApplication.objects.create(**validated_data)

        # Save photo
        if photo_file:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            resource = Resource.objects.create(
                file_path=photo_file,
                relation_type=content_type,
                relation_id=application.id,
            )
            application.photo = resource
            application.save(update_fields=["photo"])

        # Save uploaded files
        for file in files:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            Resource.objects.create(
                file_path=file,
                relation_type=content_type,
                relation_id=application.id,
            )

        return application

    def update(self, instance, validated_data):
        files = validated_data.pop("files", [])
        photo_file = validated_data.pop("photo_file", None)
        domain_name = validated_data.pop("domain_name", None)
        sub_domain_name = validated_data.pop("sub_domain_name", None)

        # Domain logic
        if domain_name:
            domain = Domain.objects.filter(name=domain_name, status="approved").first()
            if not domain:
                domain = Domain.objects.create(name=domain_name, status="pending")
            validated_data["domain"] = domain

        # SubDomain logic
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

        # Update instance
        instance = super().update(instance, validated_data)

        # Save photo
        if photo_file:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            resource = Resource.objects.create(
                file_path=photo_file,
                relation_type=content_type,
                relation_id=instance.id,
            )
            instance.photo = resource
            instance.save(update_fields=["photo"])

        # Save additional files
        for file in files:
            content_type = ContentType.objects.get_for_model(ConsultantApplication)
            Resource.objects.create(
                file_path=file,
                relation_type=content_type,
                relation_id=instance.id,
            )

        return instance
