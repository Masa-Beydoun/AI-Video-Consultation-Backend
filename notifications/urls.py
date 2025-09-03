from django.urls import path
from .views import register_device_token


urlpatterns = [
    path("register/", register_device_token, name="register_device_token"),

]
