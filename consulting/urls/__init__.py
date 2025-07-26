from django.urls import path, include

urlpatterns = [
    path('domains/', include('consulting.urls.domain_urls')),
    path('subdomains/', include('consulting.urls.subdomain_urls')),
    # path('consultations/', include('consulting.urls.consultation_urls')),
]
