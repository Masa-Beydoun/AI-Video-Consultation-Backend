# # consulting/models/email_verification.py
# from django.db import models
# from django.utils import timezone
# from datetime import timedelta

# class EmailVerificationCode(models.Model):
#     user = models.ForeignKey('User', on_delete=models.CASCADE)
#     code = models.CharField(max_length=4)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def is_expired(self):
#         return self.created_at < timezone.now() - timedelta(minutes=10)
