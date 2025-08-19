from rest_framework import serializers
from consulting.models.favorite import Favorite
from consulting.models.consultant import Consultant
from authentication.serializers import User

class FavoriteSerializer(serializers.ModelSerializer):
    # Remove user_id from being supplied externally
    consultant_id = serializers.PrimaryKeyRelatedField(
        queryset=Consultant.objects.all(), source='consultant'
    )

    class Meta:
        model = Favorite
        fields = ['id', 'consultant_id', 'added_at']
        read_only_fields = ['id', 'added_at']

    def validate(self, attrs):
        user = self.context['request'].user  # use authenticated user
        consultant = attrs['consultant']

        if consultant.user == user:
            raise serializers.ValidationError("A consultant cannot favorite themselves.")

        return attrs

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user  # set the user automatically
        favorite, created = Favorite.objects.get_or_create(**validated_data)
        if not created:
            raise serializers.ValidationError("This favorite already exists.")
        return favorite

    def remove(self):
        user = self.context['request'].user
        consultant = self.validated_data['consultant']

        try:
            favorite = Favorite.objects.get(user=user, consultant=consultant)
            favorite.delete()
            return True
        except Favorite.DoesNotExist:
            raise serializers.ValidationError("This favorite does not exist.")
