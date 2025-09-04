from django.conf import settings
from django.db import models
from .chat import Chat


SENDER_CHOICES = [
        ('U', 'User'),
        ('C', 'Consultant'),
    ]

class Message(models.Model):
    id = models.AutoField(primary_key=True)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)

    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    text = models.TextField()
    summary = models.TextField(blank=True, null=True)
    sent_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return " "
