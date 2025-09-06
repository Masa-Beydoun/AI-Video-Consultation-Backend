from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from .firebase import init_firebase
from firebase_admin import messaging
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import DeviceToken
from django.contrib.auth import get_user_model

from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Notification
from .serializers import NotificationSerializer

@csrf_exempt
@require_POST
def register_device_token(request):
    """
    Save the device token sent by frontend at login (form data)
    """
    try:
        token = request.POST.get("token")
        user_id = request.POST.get("user_id")  # or get from authenticated user

        if not token or not user_id:
            return JsonResponse({"ok": False, "error": "Missing token or user_id"}, status=400)

        User = get_user_model()
        user = User.objects.get(id=user_id)

        # Update or create to avoid duplicates
        DeviceToken.objects.update_or_create(user=user, defaults={"token": token})
        return JsonResponse({"ok": True})

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)

class NotificationViewSet(viewsets.ModelViewSet):
    serializer_class = NotificationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Notification.objects.filter(user=self.request.user).order_by("-created_at")

    @action(detail=True, methods=["post"])
    def mark_as_read(self, request, pk=None):
        notification = self.get_object()
        notification.read = True
        notification.save(update_fields=["read"])
        return Response({"ok": True, "status": "read"})