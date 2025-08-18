from django.urls import path, include
from rest_framework.routers import DefaultRouter
from consulting.views.consultant_application_views import ConsultantApplicationViewSet

router = DefaultRouter()
router.register(r'', ConsultantApplicationViewSet, basename='consultant-application')

urlpatterns = [
    path('', include(router.urls)),
]
