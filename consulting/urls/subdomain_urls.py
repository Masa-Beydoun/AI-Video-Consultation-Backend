# consulting/urls/subdomain_urls.py
from rest_framework.routers import DefaultRouter
from consulting.views.subdomain_views import SubDomainViewSet

router = DefaultRouter()
router.register(r'', SubDomainViewSet, basename='subdomain')

urlpatterns = router.urls
