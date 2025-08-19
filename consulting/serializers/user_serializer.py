# # serializers/user_serializer.py
# from rest_framework import serializers
# from consulting.models import User

# class UserRegisterSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = ['email', 'first_name', 'last_name', 'phone_number', 'password', 'role', 'gender']
#         extra_kwargs = {'password': {'write_only': True}}
