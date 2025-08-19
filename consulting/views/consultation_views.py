# consultations/views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from consulting.models import Consultation
from consulting.serializers import ConsultationSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from consulting.models import Resource
from django.contrib.contenttypes.models import ContentType

class ConsultationViewSet(viewsets.ViewSet):
    parser_classes = [MultiPartParser, FormParser]  # Allow file upload

    def create(self, request):
        # Step 1: Extract file (optional)
        file = request.FILES.get('resource.file_path', None)

        # Step 2: Create Consultation first
        serializer = ConsultationSerializer(data=request.data)
        if serializer.is_valid():
            consultation = serializer.save()

            # Step 3: If file exists, create Resource linked to consultation
            if file:
                Resource.objects.create(
                    file_path=file,
                    relation_type=ContentType.objects.get_for_model(consultation),
                    relation_id=consultation.id,
                )

            return Response(ConsultationSerializer(consultation).data, status=201)
        
        return Response(serializer.errors, status=400)
    def retrieve(self, request, pk=None):
        try:
            consultation = Consultation.objects.get(pk=pk)
            serializer = ConsultationSerializer(consultation)
            return Response(serializer.data)
        except Consultation.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)

    def update(self, request, pk=None):
        try:
            consultation = Consultation.objects.get(pk=pk)
        except Consultation.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)

        serializer = ConsultationSerializer(consultation, data=request.data, partial=True)
        if serializer.is_valid():
            updated = serializer.save()
            return Response(ConsultationSerializer(updated).data)
        return Response(serializer.errors, status=400)

    def destroy(self, request, pk=None):
        try:
            consultation = Consultation.objects.get(pk=pk)
            consultation.delete()
            return Response(status=204)
        except Consultation.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)

    @action(detail=False, methods=['get'], url_path='by-consultant/(?P<consultant_id>[^/.]+)')
    def by_consultant(self, request, consultant_id=None):
        consultations = Consultation.objects.filter(consultant_id=consultant_id)
        serializer = ConsultationSerializer(consultations, many=True)
        return Response(serializer.data)
