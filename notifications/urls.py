from django.urls import path
from .views import register_device_token
from rest_framework.routers import DefaultRouter
from .views import NotificationViewSet
from django.urls import path, include


router = DefaultRouter()
router.register(r"", NotificationViewSet, basename="notification")


urlpatterns = [
    path("register/", register_device_token, name="register_device_token"),
    path("", include(router.urls)),   # 👈 this line is missing


]
