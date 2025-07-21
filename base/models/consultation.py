from django.db import models
from django.utils import timezone

class Consultation(models.Model):
    id = models.AutoField(primary_key=True)

    consultant = models.ForeignKey(
        'Consultant',
        on_delete=models.CASCADE,
        related_name='consultations'
    )

    text = models.TextField()

    domain = models.CharField( max_length=50)
    question = models.CharField(max_length=255)
    number_of_used = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    # resource id

    def __str__(self):
        return f"Consultation by {self.consultant.user.first_name} - {self.question[:30]}"
