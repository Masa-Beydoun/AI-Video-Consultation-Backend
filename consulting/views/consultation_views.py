# consultations_views.py
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
from notifications.firebase import send_notification_to_user  # adjust to your actual notification util

from chat.models.waitingquestion import WaitingQuestion
import tempfile, os, mimetypes
import numpy as np
from chat.Chat_AI.full_matching import match_question
from sentence_transformers import SentenceTransformer

# Load model once (e.g., in your viewset class or globally)
faq_model = SentenceTransformer("all-MiniLM-L6-v2")

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
                consultant = getattr(request.user, "consultant_profile", None)
                reference_path = None

                # Handle reference image path
                if consultant and consultant.photo:
                    try:
                        if hasattr(consultant.photo, 'file_path'):
                            reference_path = consultant.photo.file_path.path
                        elif hasattr(consultant.photo, 'path'):
                            reference_path = consultant.photo.path
                        elif hasattr(consultant.photo, 'url'):
                            reference_path = consultant.photo.url
                        if reference_path and not os.path.exists(reference_path):
                            print(f"[WARN] Reference image not found at {reference_path}")
                            reference_path = None
                    except Exception as e:
                        print(f"[WARN] Could not access reference image: {e}")
                        reference_path = None

                results = run_all_checks(file_path, reference_image_path=reference_path)
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

            # Updated quality assessment logic
            failed_checks = []
            for check_name, check_result in results.items():
                if isinstance(check_result, dict):
                    status = check_result.get("status", "")
                    if "Too" in str(status) or "not" in str(status).lower() or "error" in check_result:
                        failed_checks.append(check_name)
                    if check_name == "identity_verification":
                        if not check_result.get("verified", False) and "error" not in check_result:
                            failed_checks.append("identity_not_verified")

            passed = len(failed_checks) == 0

            qc.quality_report = results
            qc.checked_at = timezone.now()
            qc.status = "approved" if passed else "rejected"
            qc.save()

            response_data = {
                "status": "approved" if passed else "rejected",
                "resource_id": resource.id,
                "quality_check_id": qc.id,
                "results": results,
            }

            if failed_checks:
                response_data["failed_checks"] = failed_checks

            if passed:
                return Response(response_data, status=201)
            else:
                resource.delete()
                return Response(response_data, status=400)

        finally:
            # Ensure file is closed and removed safely
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"[WARN] Could not delete temp file {file_path}, still in use.")

    # -------------------------------
    # 3. Segmentation endpoint
    # -------------------------------
    @action(detail=False, methods=["post"], url_path="segment-from-quality")
    def segment_from_quality(self, request):
        qc_id = request.data.get("quality_check_id")
        if not qc_id:
            return Response({"error": "quality_check_id is required"}, status=400)

        try:
            qc = ResourceQualityCheck.objects.select_related("resource").get(pk=qc_id)
        except ResourceQualityCheck.DoesNotExist:
            return Response({"error": "quality_check_id not found"}, status=404)

        resource = qc.resource

        # Resolve file path
        def _resource_file_path(resource):
            file_field = getattr(resource, "file", None) or getattr(resource, "file_path", None)
            if hasattr(file_field, "path"):
                return file_field.path
            if isinstance(file_field, str) and file_field:
                return file_field
            return None

        file_path = _resource_file_path(resource)
        if not file_path:
            return Response({"error": "Could not resolve file path"}, status=500)

        # 🔹 Call your segmentation utility
        try:
            segments = segment_video_into_consultations(file_path, model_dir="./qa_classifier")
            if not segments:
                return Response({"error": "No Q/A segments detected"}, status=400)

            created_consultations = []
            for seg in segments:
                question = seg.get("question") or "No question"
                answer = seg.get("answer") or "No answer"
                confidence_q = seg.get("confidence_question")
                confidence_a = seg.get("confidence_answer")

                consultation = Consultation.objects.create(
                    consultant=request.user.consultant_profile,
                    question=question,
                    answer=answer,
                    consultation_type="video",
                    resource_id=resource.id,  # <--- Link the resource!
                    confidence_question=confidence_q,
                    confidence_answer=confidence_a,
                )
            
                # Update resource's relation_id to this consultation
                resource.relation_id = consultation.id
                resource.save(update_fields=["relation_id"])

                created_consultations.append(consultation)


            return Response(
                {
                    "ok": True,
                    "quality_check_id": qc.id,
                    "resource_id": resource.id,
                    "consultations": ConsultationSerializer(created_consultations, many=True).data,
                    "segments": segments,
                },
                status=201,
            )

        except Exception as e:
            return Response({"error": str(e)}, status=500)


    @action(detail=False, methods=["post"], url_path="answer-waiting-question")
    def answer_waiting_question(self, request):
        """
        Consultant answers a waiting question and uploads a consultation.
        """
        waiting_id = request.data.get("waiting_question_id")
        question = request.data.get("question")
        answer = request.data.get("answer")

        if not waiting_id:
            return Response({"error": "waiting_question_id is required"}, status=400)

        try:
            waiting = WaitingQuestion.objects.select_related("user", "consultant").get(pk=waiting_id)
        except WaitingQuestion.DoesNotExist:
            return Response({"error": "WaitingQuestion not found"}, status=404)

        # prevent duplicate answers
        if waiting.answered:
            return Response({"error": "This question has already been answered"}, status=400)

        # ensure consultant is the same as the one assigned
        if waiting.consultant != request.user.consultant_profile:
            return Response({"error": "You cannot answer this question"}, status=403)

        if not answer:
            return Response({"error": "Answer is required"}, status=400)

        # create Consultation using question from WaitingQuestion
        consultation = Consultation.objects.create(
            consultant=request.user.consultant_profile,
            question=waiting.question if not question else question,
            answer=answer,
            consultation_type="text",  # or detect from context
        )

        # mark WaitingQuestion as answered
        waiting.answered = True
        waiting.save()

        send_notification_to_user(
            waiting.user,
            title="Your question has been answered",
            body=f"Consultant {request.user.consultant_profile.user.get_username()} uploaded an answer."
        )

        return Response(ConsultationSerializer(consultation).data, status=201)


    @action(
        detail=False,
        methods=["post"],
        url_path="segment-answer-waiting",
        parser_classes=[MultiPartParser, FormParser],
    )
    def segment_answer_waiting(self, request):
        """
        Consultant uploads a video/audio file,
        system segments it into Q/A, and answers a waiting question.
        """
        waiting_id = request.data.get("waiting_question_id")
        uploaded_file = request.FILES.get("file")

        if not waiting_id:
            return Response({"error": "waiting_question_id is required"}, status=400)
        if not uploaded_file:
            return Response({"error": "'file' is required"}, status=400)

        # fetch waiting question
        try:
            waiting = WaitingQuestion.objects.select_related("user", "consultant").get(pk=waiting_id)
        except WaitingQuestion.DoesNotExist:
            return Response({"error": "WaitingQuestion not found"}, status=404)

        # prevent duplicate answers
        if waiting.answered:
            return Response({"error": "This question has already been answered"}, status=400)

        # ensure consultant is the same as the one assigned
        if waiting.consultant != request.user.consultant_profile:
            return Response({"error": "You cannot answer this question"}, status=403)

        # save uploaded file temporarily
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        for chunk in uploaded_file.chunks():
            tmp_file.write(chunk)
        tmp_file.flush()
        tmp_file.close()
        file_path = tmp_file.name

        try:
            # segment file into Q/A pairs
            segments = segment_video_into_consultations(file_path, model_dir="./qa_classifier")

            if not segments:
                return Response({"error": "No Q/A segments detected"}, status=400)


            # faqs = [{"consultation_id": i, "question": seg.get("question", ""), "answer": seg.get("answer", "")}
            #         for i, seg in enumerate(segments)]
            # faq_embeddings = faq_model.encode([f["question"] for f in faqs], convert_to_tensor=True)

            # match = match_question(waiting.question, faq_model, faqs, faq_embeddings, threshold=0.65)

            # if not match["matched"]:
            #     return Response(
            #         {"error": "Uploaded consultation does not match the user’s question"},
            #         status=400,
            #     )



            created_consultations = []
            for seg in segments:
                question = seg.get("question") or waiting.question
                answer = seg.get("answer")
                if not answer:
                    continue

                consultation = Consultation.objects.create(
                    consultant=request.user.consultant_profile,
                    question=question,
                    answer=answer,
                    consultation_type="video",  # since we segmented from file
                )
                created_consultations.append(consultation)

            if not created_consultations:
                return Response({"error": "No valid consultations created"}, status=400)

            # mark waiting question answered
            waiting.answered = True
            waiting.save()

            send_notification_to_user(
                waiting.user,
                title="Your question has been answered",
                body=f"Consultant {request.user.consultant_profile.user.get_full_name()} uploaded an answer."
            )


            return Response(
                {
                    "ok": True,
                    "waiting_question_id": waiting.id,
                    "consultations": ConsultationSerializer(created_consultations, many=True).data,
                    "segments": segments,
                },
                status=201,
            )

        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass

    @action(detail=False, methods=["get"], url_path="my-consultations")
    def list_my_consultations(self, request):
        """List all consultations for the authenticated consultant"""
        consultant = getattr(request.user, "consultant_profile", None)
        if not consultant:
            return Response({"error": "You are not a consultant"}, status=status.HTTP_403_FORBIDDEN)

        consultations = Consultation.objects.filter(consultant=consultant)
        serializer = ConsultationSerializer(consultations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=["delete"], url_path="delete")
    def delete_consultation(self, request, pk=None):
        """Delete a consultation if owned by the consultant"""
        consultant = getattr(request.user, "consultant_profile", None)
        if not consultant:
            return Response({"error": "You are not a consultant"}, status=status.HTTP_403_FORBIDDEN)

        try:
            consultation = Consultation.objects.get(pk=pk, consultant=consultant)
        except Consultation.DoesNotExist:
            return Response({"error": "Consultation not found or you don’t have permission"},
                            status=status.HTTP_404_NOT_FOUND)

        consultation.delete()
        return Response({"message": "Consultation deleted successfully"}, status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=["get"], url_path="top-10")
    def top_consultations(self, request):
        """Get top 10 consultations by views_count"""
        consultations = Consultation.objects.all().order_by('-views_count')[:10]
        serializer = ConsultationSerializer(consultations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
