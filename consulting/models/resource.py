from django.db import models
from django.utils import timezone
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
import os
import mimetypes

def resource_file_path(instance, filename):
    # Get the content type model name (e.g., "consultation")
    model_name = instance.relation_type.model if instance.relation_type else 'unknown'
    return os.path.join('resources', model_name, filename)

class Resource(models.Model):
    id = models.AutoField(primary_key=True)

    file_path = models.FileField(upload_to='user_consultation_files/')
    file_meta_data = models.JSONField(blank=True, null=True)

    # Generic relation
    relation_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, null=True, blank=True)
    relation_id = models.PositiveIntegerField(null=True, blank=True)
    related_object = GenericForeignKey('relation_type', 'relation_id')

    created_at = models.DateTimeField(default=timezone.now)



    def save(self, *args, **kwargs):
        if self.file_path and not self.file_meta_data:
            mime_type, _ = mimetypes.guess_type(self.file_path.name)
            self.file_meta_data = {
                "file_name": self.file_path.name,
                "file_size_bytes": self.file_path.size,
                "file_type": mime_type or "unknown"
            }
        super().save(*args, **kwargs)

    def __str__(self):
        return f"File for {self.relation_type} ID {self.relation_id}"
