# consulting/urls/video_quality_urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from consulting.views.video_quality_views import VideoQualityViewSet

router = DefaultRouter()
router.register(r'', VideoQualityViewSet, basename='video-quality')

urlpatterns = router.urls
