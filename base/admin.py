from django.contrib import admin

# Register your models here.
from consulting.models.user import User
from consulting.models.favorite import Favorite
from consulting.models.userConsultation import UserConsultation

from consulting.models.consultant import Consultant
from consulting.models.registrationRequests import RegisterationRequests

from consulting.models.consultation import Consultation

from consulting.models.domain import Domain
from consulting.models.subdomain import SubDomain


from consulting.models.resource import Resource

admin.site.register(User)
admin.site.register(Favorite)
admin.site.register(UserConsultation)
admin.site.register(Consultant)
admin.site.register(RegisterationRequests)
admin.site.register(Consultation)
admin.site.register(Domain)
admin.site.register(SubDomain)
admin.site.register(Resource)

