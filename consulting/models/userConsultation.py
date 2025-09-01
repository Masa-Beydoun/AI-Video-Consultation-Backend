from django.db import models
from django.utils import timezone
from .consultant import Consultant
from .user import User

class UserConsultation(models.Model):
    id = models.AutoField(primary_key=True)

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='user_consultations'
    )

    # rating by consultant 
    consultation = models.ForeignKey(
        Consultant,
        on_delete=models.CASCADE,
        related_name='user_interactions'
    )

    date = models.DateTimeField(default=timezone.now)

    rate = models.PositiveSmallIntegerField(
        default=0,
        choices=[(i, i) for i in range(6)],  # 0 to 5
        help_text="Rating from 0 (worst) to 5 (best)"
    )

    class Meta:
        unique_together = ('user', 'consultation')  # Each user rates a consultation once

    def __str__(self):
        return f"{self.user.email} rated {self.consultation.id} as {self.rate}"
