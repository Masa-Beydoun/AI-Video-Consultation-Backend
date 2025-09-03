# notifications/firebase.py
from django.conf import settings
from firebase_admin import credentials, initialize_app, get_app
import firebase_admin

def init_firebase():
    try:
        get_app()  # check if already initialized
    except ValueError:
        cred = credentials.Certificate(settings.FIREBASE_CONFIG_PATH)
        initialize_app(cred)
