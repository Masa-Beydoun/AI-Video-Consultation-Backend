# consulting/urls/consultant_urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from consulting.views.consultant_views import ConsultantViewSet  # adjust import if needed

router = DefaultRouter()
router.register(r'', ConsultantViewSet, basename='consultant')

urlpatterns = [
    path('', include(router.urls)),
]
