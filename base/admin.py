from django.contrib import admin
# Register your models here.
from consulting.models.__init__ import Resource, SubDomain, Domain, Consultation,Review,ResourceQualityCheck
from consulting.models.user import  User
from consulting.models.__init__ import ConsultantApplication, Consultant, UserConsultation, Favorite


admin.site.register(User)
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


