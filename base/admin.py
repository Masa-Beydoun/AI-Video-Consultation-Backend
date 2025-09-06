from django.contrib import admin
# Register your models here.
from consulting.models.__init__ import Resource, SubDomain, Domain, Consultation,Review,ResourceQualityCheck
from consulting.models.user import  User
from consulting.models.__init__ import ConsultantApplication, Consultant, UserConsultation, Favorite

from chat.models.chat import Chat
from chat.models.message import Message
from chat.models.messageresource import MessageResource
from chat.models.waitingquestion import WaitingQuestion
from notifications.models import DeviceToken,Notification


admin.site.register(User)
admin.site.register(DeviceToken)
admin.site.register(Notification)
admin.site.register(Chat)
admin.site.register(MessageResource)
admin.site.register(WaitingQuestion)
admin.site.register(Message)
admin.site.register(Favorite)
admin.site.register(UserConsultation)
admin.site.register(Consultant)
admin.site.register(ConsultantApplication)
admin.site.register(Consultation)
admin.site.register(Domain)
admin.site.register(SubDomain)
admin.site.register(Resource)
admin.site.register(Review)
admin.site.register(ResourceQualityCheck)


