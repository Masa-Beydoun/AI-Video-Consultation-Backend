from django.conf import settings
from django.db import models

import sys
sys.path.append("../../")
from consulting.models.user import User
from consulting.models.consultant import Consultant


class Chat(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="user")
    consultant = models.ForeignKey(Consultant, on_delete=models.CASCADE, related_name="consultant")

    title = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return " "
    


