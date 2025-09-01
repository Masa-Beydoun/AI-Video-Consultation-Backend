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
from consulting.utils.segmenter_service import segment_video_into_consultations
from django.core.files import File


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



    @action(detail=True, methods=['post'])
    def segment(self, request, pk=None):
        # Get approved resource by id
        try:
            resource = Resource.objects.get(id=pk)
            qc = resource.quality_check
            if qc.status != "approved":
                return Response({"error": "Video not approved for segmentation"}, status=400)
        except (Resource.DoesNotExist, ResourceQualityCheck.DoesNotExist):
            return Response({"error": "Resource or quality check not found"}, status=404)

        temp_files = []
        try:
            # Save original video temporarily
            tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resource.file_path.name)[1])
            for chunk in resource.file_path.chunks():
                tmp_video.write(chunk)
            tmp_video.flush()
            tmp_video.close()
            video_path = tmp_video.name
            temp_files.append(video_path)

            # --- Run segmentation util ---
            from consulting.utils.segmenter_service import segment_video_into_consultations
            segments = segment_video_into_consultations(video_path)  # list of paths

            # Get consultant
            try:
                consultant = request.user.consultant
            except Consultant.DoesNotExist:
                return Response({"error": "User is not linked to a consultant"}, status=400)

            # Save segments
            created_consults = []
            for seg_path in segments:
                with open(seg_path, "rb") as f:
                    seg_resource = Resource.objects.create(
                        file_path=File(f, name=os.path.basename(seg_path)),
                        relation_type="consultation_segment",
                        relation_id=0
                    )
                consultation = Consultation.objects.create(
                    consultant=consultant,
                    resource=seg_resource
                )
                created_consults.append(consultation.id)

            # (Optional) delete or mark the original resource
            resource.delete()

            return Response({
                "status": "segmented and stored successfully",
                "consultations": created_consults
            })
        finally:
            for f in temp_files:  # only delete the original tmp video
                try:
                    os.remove(f)
                except:
                    pass
