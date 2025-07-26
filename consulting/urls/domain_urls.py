from django.urls import path
from ..views import domain_views

urlpatterns = [
    path('', domain_views.get_all_domains, name='get_all_domains'),
    path('<int:pk>/', domain_views.get_domain, name='get_domain'),
    path('create/', domain_views.create_domain, name='create_domain'),
    path('<int:pk>/update/', domain_views.update_domain, name='update_domain'),
    path('<int:pk>/delete/', domain_views.delete_domain, name='delete_domain'),
]
