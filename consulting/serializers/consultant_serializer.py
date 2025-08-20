from rest_framework import serializers
from consulting.models.consultant import Consultant
from consulting.models.favorite import Favorite
from consulting.models.user import User


class ConsultantSerializer(serializers.ModelSerializer):
    validated_by = serializers.StringRelatedField(read_only=True)
    isFavorite = serializers.SerializerMethodField()

    # pull fields from the related user
    first_name = serializers.CharField(source="user.first_name", read_only=True)
    last_name = serializers.CharField(source="user.last_name", read_only=True)
    email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        model = Consultant
        fields = [
            'id', 'first_name', 'last_name', 'email',
            'location', 'description', 'title', 'years_experience',
            'cost', 'domain', 'sub_domain',
            'validated', 'validated_by', 'validated_at',
            'added_at', 'rating', 'review_count', 'isFavorite'
        ]
        read_only_fields = fields  # all are read-only in this API

    def get_isFavorite(self, obj):
        user = self.context.get('request').user
        if not user or user.is_anonymous:
            return False
        return Favorite.objects.filter(user=user, consultant=obj).exists()

    def create(self, validated_data):
        validated_data['validated'] = False
        return super().create(validated_data)

    def update(self, instance, validated_data):
        for field in self.Meta.read_only_fields:
            validated_data.pop(field, None)
        return super().update(instance, validated_data)
