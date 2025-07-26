from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from consulting.models.domain import Domain
from consulting.serializers.domain_serializer import DomainSerializer

@api_view(['GET'])
def get_all_domains(request):
    domains = Domain.objects.all()
    serializer = DomainSerializer(domains, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def get_domain(request, pk):
    try:
        domain = Domain.objects.get(pk=pk)
    except Domain.DoesNotExist:
        return Response({'error': 'Domain not found'}, status=404)
    serializer = DomainSerializer(domain)
    return Response(serializer.data)

@api_view(['POST'])
def create_domain(request):
    serializer = DomainSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)

@api_view(['PUT'])
def update_domain(request, pk):
    try:
        domain = Domain.objects.get(pk=pk)
    except Domain.DoesNotExist:
        return Response({'error': 'Domain not found'}, status=404)
    serializer = DomainSerializer(domain, data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=400)

@api_view(['DELETE'])
def delete_domain(request, pk):
    try:
        domain = Domain.objects.get(pk=pk)
    except Domain.DoesNotExist:
        return Response({'error': 'Domain not found'}, status=404)
    domain.delete()
    return Response(status=204)
