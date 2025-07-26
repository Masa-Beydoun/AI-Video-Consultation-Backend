
from django.contrib import admin
from django.urls import path
from django.http import HttpResponse
from django.urls import path, include



def home(request):
    return HttpResponse('Home page')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('consulting.urls')),  # <-- include this, not domain_urls directly

]
