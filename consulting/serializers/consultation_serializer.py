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
            "views_count",  # new field

        ]
        read_only_fields = ["id","views_count" ]
