# consulting/views/domain_views.py

from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from consulting.models.domain import Domain
from consulting.serializers.domain_serializer import DomainSerializer
from consulting.permissions import IsAdminOrReadOnly # import your permission

class DomainViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated & IsAdminOrReadOnly]

    def list(self, request):
        domains = Domain.objects.all()
        serializer = DomainSerializer(domains, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        try:
            domain = Domain.objects.get(pk=pk)
        except Domain.DoesNotExist:
            return Response({'error': 'Domain not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = DomainSerializer(domain)
        return Response(serializer.data)

    def create(self, request):
        serializer = DomainSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):

        try:
            domain = Domain.objects.get(pk=pk)
        except Domain.DoesNotExist:
            return Response({'error': 'Domain not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = DomainSerializer(domain, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        try:
            domain = Domain.objects.get(pk=pk)
        except Domain.DoesNotExist:
            return Response({'error': 'Domain not found'}, status=status.HTTP_404_NOT_FOUND)
        domain.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
