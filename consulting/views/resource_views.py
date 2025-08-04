from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.contenttypes.models import ContentType
from django.shortcuts import get_object_or_404

from ..models import Resource
from ..serializers import ResourceSerializer

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class TestUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        print("FILES:", request.FILES)
        print("DATA:", request.data)
        return Response({"received": bool(request.FILES)}, status=200)



class ResourceCreateView(APIView):
    def post(self, request):
        print("Request received:", request.FILES, request.POST)
        print("FILES:", request.FILES)
        print("DATA:", request.data)

        serializer = ResourceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ResourceListView(APIView):
    def get(self, request):
        resources = Resource.objects.all()
        serializer = ResourceSerializer(resources, many=True, context={'request': request})
        return Response(serializer.data)


class ResourceDetailView(APIView):
    def get(self, request, pk):
        resource = get_object_or_404(Resource, pk=pk)
        serializer = ResourceSerializer(resource, context={'request': request})
        return Response(serializer.data)

    def put(self, request, pk):
        resource = get_object_or_404(Resource, pk=pk)
        serializer = ResourceSerializer(resource, data=request.data, partial=True, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        resource = get_object_or_404(Resource, pk=pk)
        resource.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ResourceByRelationView(APIView):
    def get(self, request, model_name, relation_id):
        try:
            content_type = ContentType.objects.get(model=model_name)
        except ContentType.DoesNotExist:
            return Response({"error": "Invalid model name"}, status=400)

        resources = Resource.objects.filter(
            relation_type=content_type,
            relation_id=relation_id
        )
        serializer = ResourceSerializer(resources, many=True, context={'request': request})
        return Response(serializer.data)
