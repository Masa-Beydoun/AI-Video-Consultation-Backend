# consultations/views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from django.conf import settings
from django.core.files import File
from django.utils import timezone
from django.contrib.contenttypes.models import ContentType

from consulting.models.consultation import Consultation
from consulting.models.resource import Resource
from consulting.models.resource_quality_check import ResourceQualityCheck
from consulting.serializers.consultation_serializer import ConsultationSerializer
from consulting.permissions import IsConsultant
from consulting.utils.file_checks import run_all_checks, run_audio_checks
from consulting.utils.segmenter_service import segment_video_into_consultations

import tempfile, os, mimetypes
import numpy as np


class ConsultationViewSet(viewsets.ModelViewSet):
    queryset = Consultation.objects.all()
    serializer_class = ConsultationSerializer
    permission_classes = [IsConsultant]

    # -------------------------------
    # 1. Default create: Q/A pair only
    # -------------------------------
    def create(self, request):
        question = request.data.get("question")
        answer = request.data.get("answer")

        if not (question and answer):
            return Response(
                {"error": "You must provide both 'question' and 'answer'"},
                status=400,
            )

        consultation = Consultation.objects.create(
            consultant=request.user.consultant_profile,
            question=question,
            answer=answer,
            consultation_type="text",
        )
        return Response(ConsultationSerializer(consultation).data, status=201)

    # -------------------------------
    # 2. Upload endpoint (video/audio)
    # -------------------------------
    @action(
        detail=False,
        methods=["post"],
        url_path="check-quality",
        parser_classes=[MultiPartParser, FormParser],
    )
    def upload(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "'file' is required"}, status=400)

        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        if mime_type and mime_type.startswith("video"):
            file_kind = "video"
        elif mime_type and mime_type.startswith("audio"):
            file_kind = "audio"
        else:
            return Response({"error": "Unsupported file type"}, status=400)

        # Save resource with proper ContentType
        resource = Resource.objects.create(
            file_path=uploaded_file,
            relation_type=ContentType.objects.get_for_model(Consultation),
            relation_id=0,
        )

        # Create QC object and store type
        qc = ResourceQualityCheck.objects.create(resource=resource)

        # Save temp file for checks
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(resource.file_path.name)[1]
        )
        for chunk in resource.file_path.chunks():
            tmp_file.write(chunk)
        tmp_file.flush()
        tmp_file.close()
        file_path = tmp_file.name

        try:
            # Run appropriate checks
            if file_kind == "video":
                results = run_all_checks(file_path)
            else:
                results = run_audio_checks(file_path)

            # Convert numpy types to native Python
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(x) for x in obj]
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                else:
                    return obj

            results = convert_numpy_types(results)
            passed = all("Too" not in str(v.get("status", "")) for v in results.values())

            qc.quality_report = results
            qc.checked_at = timezone.now()
            qc.status = "approved" if passed else "rejected"
            qc.save()

            if passed:
                return Response(
                    {
                        "status": "approved",
                        "resource_id": resource.id,
                        "quality_check_id": qc.id,
                        "results": results,
                    },
                    status=201,
                )
            else:
                resource.delete()
                return Response(
                    {
                        "status": "rejected",
                        "quality_check_id": qc.id,
                        "results": results,
                    },
                    status=400,
                )

        finally:
            os.remove(file_path)

    # -------------------------------
    # 3. Segmentation endpoint
    # -------------------------------


    @action(detail=False, methods=["post"], url_path="segment-from-quality")
    def segment_from_quality(self, request):
        """
        POST body: {"quality_check_id": <id>}
        This endpoint will find the Resource via ResourceQualityCheck and create consultations from segments.
        """
        qc_id = request.data.get("quality_check_id")
        if not qc_id:
            return Response({"error": "quality_check_id is required"}, status=400)

        try:
            qc = ResourceQualityCheck.objects.select_related("resource").get(pk=qc_id)
        except ResourceQualityCheck.DoesNotExist:
            return Response({"error": "quality_check_id not found"}, status=404)

        resource = qc.resource

        # helper to get file path from Resource (robust to 'file' or 'file_path')
        def _resource_file_path(resource):
            file_field = getattr(resource, "file", None) or getattr(resource, "file_path", None)
            if hasattr(file_field, "path"):
                return file_field.path
            if isinstance(file_field, str) and file_field:
                return file_field
            return None

        file_path = _resource_file_path(resource)
        if not file_path:
            return Response({"error": "resource has no accessible file path"}, status=400)

        consultant = getattr(resource, "consultant", None) or getattr(request.user, "consultant_profile", None)
        segments = segment_video_into_consultations(file_path)

        created_ids = []
        for seg in segments:
            with open(os.path.join(settings.MEDIA_ROOT, seg["file_path"]), "rb") as f:
                new_res = Resource.objects.create(
                    file_path=File(f, name=os.path.basename(seg["file_path"])),
                    relation_type=ContentType.objects.get_for_model(Consultation),
                    relation_id=0,  # after creating the consultation
                )


            new_consult = Consultation.objects.create(
                consultant=consultant,
                resource=new_res,
                question=seg["question"],
                answer=seg["answer"],
                start_time=seg.get("start"),
                end_time=seg.get("end"),
            )
            # Step 3: update the resource with the consultation id
            new_res.relation_id = new_consult.id
            new_res.save(update_fields=["relation_id"])
            created_ids.append(new_consult.id)

        # optionally delete original resource if desired:
        try:
            resource.delete(save=True)
        except Exception:
            pass

        return Response({"consultations": created_ids}, status=201)

    @action(detail=False, methods=["get"], url_path="my-role", permission_classes=[IsAuthenticated])
    def my_role(self, request):
        """
        Returns the role of the authenticated user.
        """
        user = request.user
        return Response({
            "user_id": user.id,
            "email": user.email,
            "role": getattr(user, "role", None),
        }, status=200)