from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from .firebase import init_firebase
from firebase_admin import messaging
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import DeviceToken
from django.contrib.auth import get_user_model


