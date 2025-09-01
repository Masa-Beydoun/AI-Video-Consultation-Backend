from rest_framework import serializers
from consulting.models.consultation import Consultation

class ConsultationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Consultation
        fields = [
            "id", "consultant", "question", "question_end",
            "answer", "answer_start", "confidence_question",
            "confidence_answer"
        ]
        read_only_fields = ["id"]
