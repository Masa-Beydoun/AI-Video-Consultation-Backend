from django.db import models
from django.utils import timezone
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
import jsonfield  # or JSONField if using Django 4.0+

class ResourceQualityCheck(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]

    resource = models.OneToOneField('Resource', on_delete=models.CASCADE, related_name='quality_check')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="pending")
    quality_report = models.JSONField(null=True, blank=True)  # store the run_all_checks result
    checked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"QualityCheck for Resource {self.resource.id} ({self.status})"
