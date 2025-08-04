# consulting/urls/domain_urls.py

from rest_framework.routers import DefaultRouter
from consulting.views.domain_views import DomainViewSet

router = DefaultRouter()
router.register(r'', DomainViewSet, basename='domain')

urlpatterns = router.urls
