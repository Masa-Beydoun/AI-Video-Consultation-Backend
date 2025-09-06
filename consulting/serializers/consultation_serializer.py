from rest_framework import serializers
from consulting.models.consultation import Consultation

class ConsultationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Consultation
        fields = [
            "id",
            "consultant",
            "question",
            "answer",
            "confidence_question",
            "confidence_answer",
        ]
        read_only_fields = ["id"]
