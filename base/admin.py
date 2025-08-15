from django.contrib import admin
# Register your models here.
from consulting.models.__init__ import Resource, SubDomain, Domain, Consultation
from consulting.models.user import  User
from consulting.models.__init__ import RegisterationRequests, Consultant, UserConsultation, Favorite


admin.site.register(User)
# admin.site.register(EmailVerificationCode)
# admin.site.register(CustomUser)
# admin.site.register(AuthToken)
# admin.site.register(VerificationCode)
admin.site.register(Favorite)
admin.site.register(UserConsultation)
admin.site.register(Consultant)
admin.site.register(RegisterationRequests)
admin.site.register(Consultation)
admin.site.register(Domain)
admin.site.register(SubDomain)
admin.site.register(Resource)


