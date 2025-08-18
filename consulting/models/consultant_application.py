from django.db import models
from django.utils import timezone
from .user import User
from .domain import Domain
from .subdomain import SubDomain

class ConsultantApplication(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name="consultant_applications"
    )
    location = models.CharField(max_length=100)
    description = models.TextField()
    cost = models.IntegerField(default=0)
    
    domain = models.ForeignKey(Domain, on_delete=models.SET_NULL, null=True, related_name='consultant_applications')
    sub_domain = models.ForeignKey(SubDomain, on_delete=models.SET_NULL, null=True, related_name='consultant_applications')

    years_experience = models.IntegerField(blank=True, null=True)
    languages = models.CharField(max_length=255, blank=True, null=True)

    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')

    admin_notes = models.TextField(blank=True, null=True)
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="reviewed_applications")
    reviewed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Application by {self.user.email} - {self.status}"
