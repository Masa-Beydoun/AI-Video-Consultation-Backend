# consulting/views/video_quality_views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from consulting.models.resource import Resource
from consulting.models.resource_quality_check import ResourceQualityCheck
from consulting.models.consultation import  Consultation
from consulting.permissions import IsConsultant
from consulting.utils.video_checks import run_all_checks
from rest_framework.parsers import MultiPartParser, FormParser
import tempfile, os
from django.utils import timezone
import numpy as np

class VideoQualityViewSet(viewsets.ViewSet):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsConsultant]

    @action(detail=False, methods=['post'])
    def upload(self, request):
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({"error": "'file' is required"}, status=400)

        # Save video to Resource
        resource = Resource.objects.create(
            file_path=uploaded_file,
            relation_type=None,
            relation_id=0
        )

        # Create a pending quality check
        ResourceQualityCheck.objects.create(resource=resource)

        return Response({"resource_id": resource.id, "status": "pending"}, status=201)

    @action(detail=True, methods=['post'])
    def quality_check(self, request, pk=None):
        # Get resource
        try:
            resource = Resource.objects.get(id=pk)
            qc = resource.quality_check
        except (Resource.DoesNotExist, ResourceQualityCheck.DoesNotExist):
            return Response({"error": "Resource or quality check not found"}, status=404)

        # Save file to temp location for analysis
        temp_files = []
        try:
            tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resource.file_path.name)[1])
            for chunk in resource.file_path:
                tmp_video.write(chunk)
            tmp_video.flush()
            tmp_video.close()
            video_path = tmp_video.name
            temp_files.append(video_path)

            # Run all checks
            results = run_all_checks(video_path)

            # Convert results to JSON-safe types
            results = make_json_safe(results)
            
            # Determine pass/fail (example: all statuses must not contain "Too")
            passed = all("Too" not in str(v.get("status", "")) for v in results.values())

            if passed:
                qc.status = "approved"
                qc.quality_report = results
                qc.checked_at = timezone.now()
                qc.save()
                return Response({"resource_id": resource.id, "status": "approved", "results": results})
            else:
                # delete video and QC
                resource.delete()
                return Response({"resource_id": pk, "status": "rejected", "results": results})
        finally:
            for f in temp_files:
                try:
                    os.remove(f)
                except:
                    pass

    @action(detail=True, methods=['post'])
    def segment(self, request, pk=None):
        # Get approved resource
        try:
            resource = Resource.objects.get(id=pk)
            qc = resource.quality_check
            if qc.status != "approved":
                return Response({"error": "Video not approved for segmentation"}, status=400)
        except (Resource.DoesNotExist, ResourceQualityCheck.DoesNotExist):
            return Response({"error": "Resource or quality check not found"}, status=404)

        # Here you would run your segmentation logic
        # Example: split video into parts and save in Consultation (pseudo code)
        # from your segmentation util import segment_video
        # segments = segment_video(resource.file_path.path)
        # for segment_file in segments:
        #     Consultation.objects.create(video_segment=segment_file, ...)

        # After segmentation, delete the original resource
        resource.delete()

        return Response({"status": "segmented and stored successfully"})
    import numpy as np

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

