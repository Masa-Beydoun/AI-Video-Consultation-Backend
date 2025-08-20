from django.db import models
from django.utils import timezone
from .domain import Domain
from .subdomain import SubDomain
from .user import User
from django.db.models import Avg

class Consultant(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="consultant_profile")

    # Basic info
    location = models.CharField(max_length=100)
    description = models.TextField()

    # Professional info
    title = models.CharField(max_length=100, blank=True, null=True)
    years_experience = models.PositiveIntegerField(blank=True, null=True)

    # Pricing & availability
    cost = models.IntegerField(default=0)

    # Domain linkage
    domain = models.ForeignKey(Domain, on_delete=models.SET_NULL, null=True, related_name='consultants')
    sub_domain = models.ForeignKey(SubDomain, on_delete=models.SET_NULL, null=True, related_name='consultants')

    # Validation
    validated = models.BooleanField(default=False)
    validated_by = models.ForeignKey(User, null=True, blank=True, related_name="validated_consultants", on_delete=models.SET_NULL)
    validated_at = models.DateTimeField(null=True, blank=True)
    added_at = models.DateTimeField(default=timezone.now)

    # Rating (optional)
    rating = models.FloatField(default=0)
    review_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.first_name} {self.user.last_name} ({self.domain.name if self.domain else 'No Domain'})"

    def update_rating(self):
        reviews = self.reviews.all()
        self.review_count = reviews.count()
        self.rating = reviews.aggregate(Avg("score"))["score__avg"] or 0
        self.save(update_fields=["review_count", "rating"])