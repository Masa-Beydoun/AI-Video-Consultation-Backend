from rest_framework.routers import DefaultRouter
from consulting.views.consultation_views import ConsultationViewSet

router = DefaultRouter()
router.register(r'', ConsultationViewSet, basename='consultation')

urlpatterns = router.urls
