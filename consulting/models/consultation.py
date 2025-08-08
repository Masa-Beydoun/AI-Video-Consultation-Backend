from django.db import models
from django.utils import timezone
from .consultant import Consultant
from .domain import Domain
from .subdomain import SubDomain


class Consultation(models.Model):
    id = models.AutoField(primary_key=True)

    consultant = models.ForeignKey(
        Consultant,
        on_delete=models.CASCADE,
        related_name='consultations'
    )

    text = models.TextField()

    domain = models.ForeignKey(
        Domain,
        on_delete=models.SET_NULL,
        null=True,
        related_name='domain'
    )


    sub_domain = models.ForeignKey(
        SubDomain,
        on_delete=models.SET_NULL,
        null=True,
        related_name='subdomain'
    ) 
    number_of_used = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    # resource id

    def __str__(self):
        return f"Consultation by {self.consultant.user.first_name} - {self.question[:30]}"
