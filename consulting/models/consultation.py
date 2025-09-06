# consulting/models/consultation.py
from django.db import models
from .consultant import Consultant
from .resource import Resource
import os

def resource_file_path(instance, filename):
    return os.path.join("consultants", str(instance.related_object.id), "consultations", filename)

class Consultation(models.Model):
    consultant = models.ForeignKey(Consultant, on_delete=models.CASCADE)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, related_name="consultations",null=True,blank=True)
    
    question = models.TextField()
    answer = models.TextField()


    confidence_question = models.FloatField(null=True,blank=True)
    confidence_answer = models.FloatField(null=True,blank=True)

    TYPE_CHOICES=[
        ('audio','Audio'),
        ('video','Video'),
        ('text','Text'),
    ]   
    consultation_type = models.CharField(choices= TYPE_CHOICES,null=True,blank=True)

    views_count = models.PositiveIntegerField(default=0)
    
