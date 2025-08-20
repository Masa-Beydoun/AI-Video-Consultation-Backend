# consulting/models/review.py
from django.db import models
from django.conf import settings
from .consultant import Consultant

class Review(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="reviews")
    consultant = models.ForeignKey(Consultant, on_delete=models.CASCADE, related_name="reviews")
    score = models.PositiveSmallIntegerField()  # 0–10

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'consultant')  # prevent duplicate reviews

    def __str__(self):
        return f"{self.user.email} → {self.consultant.user.email}: {self.score}"
