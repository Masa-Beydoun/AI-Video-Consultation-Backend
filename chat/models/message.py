from django.conf import settings
from django.db import models
from .chat import Chat

import sys
sys.path.append("../../")
from consulting.models.resource import Resource

SENDER_CHOICES = [
        ('U', 'User'),
        ('C', 'Consultant'),
    ]

class Message(models.Model):
    id = models.AutoField(primary_key=True)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, blank=True, null=True)

    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    text = models.TextField()
    sent_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return " "
