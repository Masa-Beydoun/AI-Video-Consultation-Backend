# consultations/urls.py or favorites/urls.py
from django.urls import path
from consulting.views.favorite_views import FavoriteView

urlpatterns = [
    path('', FavoriteView.as_view(), name='favorite'),
]
