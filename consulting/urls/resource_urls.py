from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from consulting.views.resource_views import (
    ResourceCreateView,
    ResourceListView,
    ResourceDetailView,
    ResourceByRelationView,
    TestUploadView
)

urlpatterns = [
    path('', ResourceListView.as_view(), name='resource-list'),
    path('create/', ResourceCreateView.as_view(), name='resource-create'),
    path('<int:pk>/', ResourceDetailView.as_view(), name='resource-detail'),
    path('by-relation/<str:model_name>/<int:relation_id>/', ResourceByRelationView.as_view(), name='resource-by-relation'),
    path('test-upload/', TestUploadView.as_view(), name='test-upload'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
