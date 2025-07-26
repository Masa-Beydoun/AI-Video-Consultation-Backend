# consulting/urls/subdomain_urls.py
from django.urls import path
from ..views import subdomain_views

urlpatterns = [
    path('', subdomain_views.get_all_subdomains, name='get_all_subdomains'),
    path('<int:pk>/', subdomain_views.get_subdomain, name='get_subdomain'),
    path('create/', subdomain_views.create_subdomain, name='create_subdomain'),
    path('<int:pk>/update/', subdomain_views.update_subdomain, name='update_subdomain'),
    path('<int:pk>/delete/', subdomain_views.delete_subdomain, name='delete_subdomain'),
    path('by-domain/<int:domain_id>/', subdomain_views.get_subdomains_by_domain, name='subdomains_by_domain'),
]
