# consultations/views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from consulting.models import Consultation
from consulting.serializers import ConsultationSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from consulting.models import Resource
from django.contrib.contenttypes.models import ContentType
import os
from consulting.models.consultant import Consultant
from consulting.utils.video_checks import run_all_checks
from consulting.permissions import IsConsultant


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

    @action(
        detail=False,
        methods=['post'],
        url_path='quality-check',
        parser_classes=[MultiPartParser, FormParser],
        permission_classes=[IsConsultant]
    )
    def quality_check(self, request):
        user = request.user
        try:
            consultant = user.consultant_profile
        except Consultant.DoesNotExist:
            return Response({"error": "User is not a consultant"}, status=400)

        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({"error": "'file' is required (multipart form-data)."}, status=400)

        temp_files = []
        try:
            # Save uploaded video to temp file
            import tempfile, os
            tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            for chunk in uploaded_file.chunks():
                tmp_video.write(chunk)
            tmp_video.flush()
            tmp_video.close()
            video_path = tmp_video.name
            temp_files.append(video_path)

            # Run video checks (no reference image needed)
            from consulting.utils.video_checks import run_all_checks
            results = run_all_checks(video_path, reference_image_path=None)

            return Response({"status": "ok", "results": results}, status=200)
        finally:
            # cleanup temp files
            for f in temp_files:
                try:
                    os.remove(f)
                except:
                    pass
