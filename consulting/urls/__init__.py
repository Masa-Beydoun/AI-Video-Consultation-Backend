from django.urls import path, include

urlpatterns = [
    path('domains/', include('consulting.urls.domain_urls')),
    path('subdomains/', include('consulting.urls.subdomain_urls')),
    path('resources/',include('consulting.urls.resource_urls')),
    path('consultations/',include('consulting.urls.consultation_urls')),
    path('favorites/',include('consulting.urls.favorite_urls'))

    # path('auth/', include('consulting.urls.auth_urls')),  # 👈 Add this

]
