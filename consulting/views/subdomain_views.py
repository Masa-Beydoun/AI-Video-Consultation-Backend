# consulting/views/subdomain_views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from consulting.models import SubDomain, Domain
from consulting.serializers.subdomain_serializer import SubDomainSerializer

@api_view(['GET'])
def get_all_subdomains(request):
    subdomains = SubDomain.objects.all()
    serializer = SubDomainSerializer(subdomains, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def get_subdomain(request, pk):
    try:
        subdomain = SubDomain.objects.get(pk=pk)
    except SubDomain.DoesNotExist:
        return Response({'error': 'SubDomain not found'}, status=404)
    
    serializer = SubDomainSerializer(subdomain)
    return Response(serializer.data)

@api_view(['POST'])
def create_subdomain(request):
    serializer = SubDomainSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)

@api_view(['PUT'])
def update_subdomain(request, pk):
    try:
        subdomain = SubDomain.objects.get(pk=pk)
    except SubDomain.DoesNotExist:
        return Response({'error': 'SubDomain not found'}, status=404)
    
    serializer = SubDomainSerializer(subdomain, data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=400)

@api_view(['DELETE'])
def delete_subdomain(request, pk):
    try:
        subdomain = SubDomain.objects.get(pk=pk)
    except SubDomain.DoesNotExist:
        return Response({'error': 'SubDomain not found'}, status=404)
    
    subdomain.delete()
    return Response({'message': 'Deleted successfully'})


@api_view(['GET'])
def get_subdomains_by_domain(request, domain_id):
    try:
        domain = Domain.objects.get(pk=domain_id)
    except Domain.DoesNotExist:
        return Response({'error': 'Domain not found'}, status=404)

    subdomains = domain.subdomains.all()  # uses related_name='subdomains'
    serializer = SubDomainSerializer(subdomains, many=True)
    return Response(serializer.data)
