from django.db import models


class Domain(models.Model):

    id = models.AutoField(primary_key=True)  # auto-incremented ID
    name = models.CharField(max_length=50,unique=True)

    def __str__(self):
        return self.name