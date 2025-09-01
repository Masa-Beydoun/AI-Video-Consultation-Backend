from django.urls import path, include

urlpatterns = [
    path('domains/', include('consulting.urls.domain_urls')),
    path('subdomains/', include('consulting.urls.subdomain_urls')),
    path('resources/',include('consulting.urls.resource_urls')),
    path('consultations/',include('consulting.urls.consultation_urls')),
    path('favorites/',include('consulting.urls.favorite_urls')),
    path('consultant-applications/',include('consulting.urls.consultant_application_urls')),
    path('consultants/',include('consulting.urls.consultant_urls')),
    path('reviews/',include('consulting.urls.review_urls')),
    path('videos/',include('consulting.urls.video_quality_urls')),

    # path('auth/', include('consulting.urls.auth_urls')),  # 👈 Add this

]
