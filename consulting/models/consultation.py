# consulting/models/consultation.py
from django.db import models
from .consultant import Consultant
from .resource import Resource
import os

def resource_file_path(instance, filename):
    return os.path.join("consultants", str(instance.related_object.id), "consultations", filename)

class Consultation(models.Model):
    consultant = models.ForeignKey(Consultant, on_delete=models.CASCADE)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, related_name="consultations")
    question = models.TextField()
    question_end = models.FloatField()
    answer = models.TextField()
    answer_start = models.FloatField()
    confidence_question = models.FloatField()
    confidence_answer = models.FloatField()
