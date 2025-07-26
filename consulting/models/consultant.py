from django.db import models
from django.utils import timezone
from .user import User
from .domain import Domain
from .subdomain import SubDomain

class Consultant(models.Model):
    id = models.AutoField(primary_key=True)

    location = models.CharField(max_length=100)
    description = models.TextField()
    added_at = models.DateTimeField(default=timezone.now)
    validated = models.BooleanField(default=False)
    # Link to User (assuming your custom User model is named 'User')
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='consultant_profile'
    )

    
    domain = models.ForeignKey(
        Domain,
        on_delete=models.SET_NULL,
        null=True,
        related_name='consultants'
    )

    sub_domain = models.ForeignKey(
        SubDomain,
        on_delete=models.SET_NULL,
        null=True,
        related_name='consultants'
    )

    

    def __str__(self):
        return f"{self.user.first_name} {self.user.last_name} ({self.domain.name if self.domain else 'No Domain'})"
