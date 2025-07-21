from django.db import models
from django.utils import timezone
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey

class Resource(models.Model):
    id = models.AutoField(primary_key=True)

    file_path = models.FileField(upload_to='user_consultation_files/')
    file_meta_data = models.JSONField(blank=True, null=True)

    # Generic relation
    relation_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    relation_id = models.PositiveIntegerField()
    related_object = GenericForeignKey('relation_type', 'relation_id')

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"File for {self.relation_type} ID {self.relation_id}"
