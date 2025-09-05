# notifications/firebase.py
from django.conf import settings
from firebase_admin import credentials, initialize_app, get_app
import firebase_admin
from firebase_admin import messaging


def init_firebase():
    try:
        get_app()  # check if already initialized
    except ValueError:
        cred = credentials.Certificate(settings.FIREBASE_CONFIG_PATH)
        initialize_app(cred)

def send_notification_to_user(user, title, body, data=None):
    """
    Send push notification to all devices of a user.
    data: optional dict with extra payload
    """
    from .models import DeviceToken
    init_firebase()
    
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