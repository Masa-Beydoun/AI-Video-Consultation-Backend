from rest_framework import serializers
from consulting.serializers.resource_serializer import ResourceSerializer
from consulting.models import Consultation,Resource
from django.contrib.contenttypes.models import ContentType

class ConsultationSerializer(serializers.ModelSerializer):
    attached_resource = serializers.SerializerMethodField()
    resource_file = serializers.FileField(write_only=True, required=False)


    class Meta:
        model = Consultation
        fields = [
            'id', 'consultant', 'text', 'domain', 'sub_domain',
            'number_of_used', 'created_at', 'resource_file', 'attached_resource'
        ]
        read_only_fields = ['id', 'created_at', 'number_of_used', 'attached_resource']



    def get_attached_resource(self, obj):
        resource = Resource.objects.filter(
            relation_type=ContentType.objects.get_for_model(Consultation),
            relation_id=obj.id
        ).first()
        if resource:
            return ResourceSerializer(resource).data
        return None

    def create(self, validated_data):
        resource_data = validated_data.pop('resource_file', None)
        consultation = Consultation.objects.create(**validated_data)

        if resource_data:
            Resource.objects.create(
                file_path=resource_data,
                relation_type=ContentType.objects.get_for_model(Consultation),
                relation_id=consultation.id,
            )
        return consultation


    def update(self, instance, validated_data):
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

    def validate(self, attrs):
        if 'consultant' not in attrs or attrs['consultant'] is None:
            # You could allow None or handle differently
            attrs['consultant'] = None
        return attrs
