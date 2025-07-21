from django.db import models
from django.contrib.auth.hashers import make_password


class User(models.Model):

    ROLE_CHOICES = [
        ('user', 'User'),
        ('consultant', 'Consultant'),
    ]

    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]
    id = models.AutoField(primary_key=True)  # auto-incremented ID
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    phone_number = models.CharField(max_length=10)
    password = models.CharField(max_length=128)  # will store hashed password
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    gender = models.CharField(max_length=6, choices=GENDER_CHOICES)



    def save(self, *args, **kwargs):
        # Hash the password before saving if it's not already hashed
        if not self.password.startswith('pbkdf2_'):  # Check if already hashed
            self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.email
