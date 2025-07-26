from rest_framework import serializers
from .models import User, Consultant, Consultation, Favorite, Domain, SubDomain, UserConsultation

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'

class ConsultantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Consultant
        fields = '__all__'

# Do the same for other models
