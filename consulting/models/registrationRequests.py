from django.db import models
from django.utils import timezone
from .consultant import Consultant

class RegisterationRequests(models.Model):
    id = models.AutoField(primary_key=True)

    consultant = models.OneToOneField(
        Consultant,
        on_delete=models.CASCADE,
        related_name='user_relations'
    )

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.email} <-> {self.consultant.user.first_name}"
    