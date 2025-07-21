from django.db import models

class SubDomain(models.Model):
    id = models.AutoField(primary_key=True)  # optional, Django adds it by default
    name = models.CharField(max_length=50)

    domain = models.ForeignKey(
        'Domain',                # or just Domain if defined above
        on_delete=models.CASCADE,
        related_name='subdomains'
    )

    def __str__(self):
        return self.name
