# notifications/firebase.py
from django.conf import settings
from firebase_admin import credentials, initialize_app, get_app
import firebase_admin
from firebase_admin import messaging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Firebase only once
if not firebase_admin._apps:
    cred_path = os.path.join(BASE_DIR, "secrets", "serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

def init_firebase():
    try:
        get_app()  # check if already initialized
    except ValueError:
        cred = credentials.Certificate(settings.FIREBASE_CONFIG_PATH)
        initialize_app(cred)

def send_notification_to_user(user, title, body, data=None):
    tokens = [t.token for t in user.devicetoken_set.all()]
    if not tokens:
        return {"ok": False, "error": "No device tokens for this user"}

    message = messaging.MulticastMessage(
        tokens=tokens,
        notification=messaging.Notification(title=title, body=body),
        data=data or {},
    )

    if hasattr(messaging, "send_each_for_multicast"):
        response = messaging.send_each_for_multicast(message)
    else:
        response = messaging.send_multicast(message)

    return {
        "ok": True,
        "success_count": response.success_count,
        "failure_count": response.failure_count,
    }
