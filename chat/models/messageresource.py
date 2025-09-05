from django.conf import settings
from django.db import models
from .message import Message

import sys
sys.path.append("../../")
from consulting.models.resource import Resource

class MessageResource(models.Model):

    id = models.AutoField(primary_key=True)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, blank=True, null=True)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, blank=True, null=True)

    def __str__(self):
        return " "