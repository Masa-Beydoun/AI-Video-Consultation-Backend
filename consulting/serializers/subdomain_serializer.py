from rest_framework import serializers
from consulting.models import SubDomain

class SubDomainSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubDomain
        fields = ['id', 'name', 'domain']
