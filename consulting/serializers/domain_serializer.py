from rest_framework import serializers
from consulting.models.domain import Domain  # Adjust path if needed

class DomainSerializer(serializers.ModelSerializer):
    class Meta:
        model = Domain
        fields = '__all__'  # or list the fields like ['id', 'name']
