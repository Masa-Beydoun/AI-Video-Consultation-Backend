from django.db import models

class Favorite(models.Model):
    id = models.AutoField(primary_key=True)

    user = models.ForeignKey(
        'User',
        on_delete=models.CASCADE,
        related_name='favorites'
    )

    consultant = models.ForeignKey(
        'Consultant',
        on_delete=models.CASCADE,
        related_name='favorited_by'
    )

    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'consultant')  # Prevent duplicate favorites

    def __str__(self):
        return f"{self.user.email} favorited {self.consultant.user.first_name}"
