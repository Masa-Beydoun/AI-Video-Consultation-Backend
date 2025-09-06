from django.urls import path
from .views import register_device_token
from rest_framework.routers import DefaultRouter
from .views import NotificationViewSet

router = DefaultRouter()
router.register(r"notifications", NotificationViewSet, basename="notification")


urlpatterns = [
    path("register/", register_device_token, name="register_device_token"),

]
