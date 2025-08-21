from rest_framework import serializers
from consulting.models.consultant import Consultant
from consulting.models.favorite import Favorite
from consulting.models.resource import Resource
from consulting.serializers.resource_serializer import ResourceSerializer

class ConsultantSerializer(serializers.ModelSerializer):
    validated_by = serializers.StringRelatedField(read_only=True)
    isFavorite = serializers.SerializerMethodField()

    # pull fields from the related user
    first_name = serializers.CharField(source="user.first_name", read_only=True)
    last_name = serializers.CharField(source="user.last_name", read_only=True)
    email = serializers.EmailField(source="user.email", read_only=True)

    # nested photo serializer
    photo = ResourceSerializer(read_only=True)

    # return domain/subdomain names
    domain_name = serializers.CharField(source="domain.name", read_only=True)
    sub_domain_name = serializers.CharField(source="sub_domain.name", read_only=True)

    class Meta:
        model = Consultant
        fields = [
            'id', 'first_name', 'last_name', 'email',
            'location', 'description', 'title', 'years_experience',
            'cost', 'domain', 'sub_domain',  # keep IDs if you need them
            'domain_name', 'sub_domain_name',  # add names
            'validated', 'validated_by', 'validated_at',
            'added_at', 'rating', 'review_count', 'isFavorite', 'photo'
        ]
        read_only_fields = fields  # all read-only

    def get_isFavorite(self, obj):
        user = self.context.get('request').user
        if not user or user.is_anonymous:
            return False
        return Favorite.objects.filter(user=user, consultant=obj).exists()
