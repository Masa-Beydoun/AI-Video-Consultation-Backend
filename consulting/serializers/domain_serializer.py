# consulting/serializers/domain_serializer.py
from rest_framework import serializers
from consulting.models.domain import Domain
from consulting.models.resource import Resource  # adjust import if needed

class DomainSerializer(serializers.ModelSerializer):
    photo = serializers.SerializerMethodField()

    class Meta:
        model = Domain
        fields = ['id', 'name', 'photo']  # add photo to fields

    def get_photo(self, obj):
        resource = Resource.objects.filter(
            relation_type__model='domain',
            relation_id=obj.id
        ).first()
        if resource and resource.file_path:
            request = self.context.get('request')
            return request.build_absolute_uri(resource.file_path.url) if request else resource.file_path.url
        return None
