# consulting/urls.py
from rest_framework.routers import DefaultRouter
from consulting.views.review_views import ReviewViewSet

router = DefaultRouter()
router.register(r'', ReviewViewSet, basename='review')

urlpatterns = [
    # ... your other routes
] + router.urls
