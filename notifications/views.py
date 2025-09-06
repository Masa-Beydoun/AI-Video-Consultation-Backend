# notifications/views.py
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model

from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

from .firebase import init_firebase
from firebase_admin import messaging
from .models import DeviceToken, Notification
from .serializers import NotificationSerializer


# ---------------- Device Token Registration ----------------
@csrf_exempt
@require_POST
def register_device_token(request):
    """
    Save the device token sent by frontend at login
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


# ---------------- Notification ViewSet ----------------
class NotificationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for listing, marking as read, and sending notifications
    """
    serializer_class = NotificationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """
        Return notifications for the authenticated user only
        """
        return Notification.objects.filter(user=self.request.user).order_by("-created_at")

    # --- List all notifications ---
    @action(detail=False, methods=["get"])
    def all(self, request):
        """
        Return all notifications for the user (read and unread)
        """
        notifications = self.get_queryset()
        serializer = self.get_serializer(notifications, many=True)
        return Response(serializer.data)

    # --- Mark a single notification as read ---
    @action(detail=True, methods=["post"])
    def mark_as_read(self, request, pk=None):
        """
        Mark a single notification as read
        """
        notification = self.get_object()
        notification.read = True
        notification.save(update_fields=["read"])
        return Response({"ok": True, "status": "read"})

    # --- Send notification utility ---
    @staticmethod
    def send_notification_to_user(user, title, body, data=None):
        """
        Save notification in DB and send via Firebase
        """
        init_firebase()

        # Save to DB
        Notification.objects.create(
            user=user,
            title=title,
            body=body,
            data=data or {}
        )

        tokens = list(DeviceToken.objects.filter(user=user).values_list("token", flat=True))
        if not tokens:
            return {"ok": False, "error": "No device tokens for this user"}

        message = messaging.MulticastMessage(
            tokens=tokens,
            notification=messaging.Notification(title=title, body=body),
            data=data or {}
        )

        response = messaging.send_multicast(message)
        return {
            "ok": True,
            "success_count": response.success_count,
            "failure_count": response.failure_count,
            "responses": [r.__dict__ for r in response.responses]
        }
