from django.conf import settings
from django.db import models

import sys
sys.path.append("../../")
from consulting.models.consultant import Consultant
from consulting.models.user import User

class WaitingQuestion(models.Model):

    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    consultant = models.ForeignKey(Consultant, on_delete=models.CASCADE)
    question = models.TextField()

    def __str__(self):
        return " "