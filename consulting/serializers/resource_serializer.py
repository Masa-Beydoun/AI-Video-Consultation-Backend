# serializers/resource_serializer.py

from rest_framework import serializers
from consulting.models.resource import Resource

class ResourceSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    class Meta:
        model = Resource
        fields = '__all__'
        read_only_fields = ['file_meta_data', 'created_at']

    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file_path and request:
            return request.build_absolute_uri(obj.file_path.url)
        return None
    def validate(self, data):
        # Optional: ensure that the relation_id exists in the specified relation_type model
        # relation_type = data.get('relation_type')
        # relation_id = data.get('relation_id')
        # if relation_type and relation_id:
        #     model_class = relation_type.model_class()
        #     if not model_class.objects.filter(id=relation_id).exists():
        #         raise serializers.ValidationError("Related object not found.")
        return data
