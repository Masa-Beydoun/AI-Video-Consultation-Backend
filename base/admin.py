from django.contrib import admin

# Register your models here.
from .models.user import User
from .models.favorite import Favorite
from .models.userConsultation import UserConsultation

from .models.consultant import Consultant
from .models.registrationRequests import RegisterationRequests

from .models.consultation import Consultation

from .models.domain import Domain
from .models.subdomain import SubDomain


from .models.resource import Resource

admin.site.register(User)
admin.site.register(Favorite)
admin.site.register(UserConsultation)
admin.site.register(Consultant)
admin.site.register(RegisterationRequests)
admin.site.register(Consultation)
admin.site.register(Domain)
admin.site.register(SubDomain)
admin.site.register(Resource)

