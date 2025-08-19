# consulting/views/subdomain_views.py

from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from consulting.models import SubDomain, Domain
from consulting.serializers.subdomain_serializer import SubDomainSerializer
from consulting.permissions import IsAdminOrReadOnly # import your permission
from rest_framework.permissions import IsAuthenticated

class SubDomainViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated & IsAdminOrReadOnly]


    def list(self, request):
        subdomains = SubDomain.objects.all()
        serializer = SubDomainSerializer(subdomains, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        try:
            subdomain = SubDomain.objects.get(pk=pk)
        except SubDomain.DoesNotExist:
            return Response({'error': 'SubDomain not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = SubDomainSerializer(subdomain)
        return Response(serializer.data)

    def create(self, request):
        serializer = SubDomainSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        try:
            subdomain = SubDomain.objects.get(pk=pk)
        except SubDomain.DoesNotExist:
            return Response({'error': 'SubDomain not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = SubDomainSerializer(subdomain, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        try:
            subdomain = SubDomain.objects.get(pk=pk)
        except SubDomain.DoesNotExist:
            return Response({'error': 'SubDomain not found'}, status=status.HTTP_404_NOT_FOUND)
        subdomain.delete()
        return Response({'message': 'Deleted successfully'}, status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['get'], url_path='by-domain/(?P<domain_id>[^/.]+)')
    def get_by_domain(self, request, domain_id=None):
        try:
            domain = Domain.objects.get(pk=domain_id)
        except Domain.DoesNotExist:
            return Response({'error': 'Domain not found'}, status=status.HTTP_404_NOT_FOUND)
        subdomains = domain.subdomains.all()  # uses related_name
        serializer = SubDomainSerializer(subdomains, many=True)
        return Response(serializer.data)
