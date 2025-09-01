# consulting/models/domain.py
from django.db import models

class Domain(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
    ]

    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    def __str__(self):
        return self.name
