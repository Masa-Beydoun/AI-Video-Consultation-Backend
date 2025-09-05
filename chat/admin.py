from django.contrib import admin
from .models import Chat
from .models import Message
from .models import MessageResource
from .models import WaitingQuestion

# Register your models here.

admin.site.register(Chat)
admin.site.register(Message)
admin.site.register(MessageResource)
admin.site.register(WaitingQuestion)