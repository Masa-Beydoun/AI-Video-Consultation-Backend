# consulting/serializers/review_serializer.py
from rest_framework import serializers
from consulting.models.review import Review

class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = ["id", "consultant", "score", "created_at"]
        read_only_fields = ["id", "created_at"]

    def validate_score(self, value):
        if value < 0 or value > 10:
            raise serializers.ValidationError("Score must be between 0 and 10.")
        return value
