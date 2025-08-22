from django.contrib import admin
from .models import Chat
from .models import Message

# Register your models here.

admin.site.register(Chat)
admin.site.register(Message)