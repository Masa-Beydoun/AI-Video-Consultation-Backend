from django.urls import path
from consulting.views.audio_transcriber import TranscriptionView

urlpatterns = [
    path('transcribe/', TranscriptionView.as_view(), name='transcribe'),
]
