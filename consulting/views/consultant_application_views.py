from rest_framework import viewsets, status, permissions
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied

from django.utils import timezone
from django.core.mail import EmailMultiAlternatives
from django.conf import settings
from django.core.files import File
from django.contrib.contenttypes.models import ContentType

import os

from consulting.models.consultant_application import ConsultantApplication
from consulting.models.resource import Resource
from consulting.models.consultant import Consultant
from consulting.serializers.consultant_application_serializer import ConsultantApplicationSerializer
from consulting.serializers.resource_serializer import ResourceSerializer


class ConsultantApplicationViewSet(viewsets.ModelViewSet):
    serializer_class = ConsultantApplicationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, "role") and user.role == "admin":
            return ConsultantApplication.objects.all().order_by('-created_at')
        return ConsultantApplication.objects.filter(user=user).order_by('-created_at')

    def get_permissions(self):
        return [permissions.IsAuthenticated()]

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        user = request.user
        if not (hasattr(user, "role") and user.role == "admin") and instance.user != user:
            raise PermissionDenied("You do not have permission to view this application.")

        serializer = self.get_serializer(instance)
        resources = Resource.objects.filter(
            relation_type__model='consultantapplication',
            relation_id=instance.id
        )
        resources_serialized = ResourceSerializer(resources, many=True, context={'request': request}).data

        data = serializer.data
        data['resources'] = resources_serialized
        return Response(data)

    def perform_create(self, serializer):
        user = self.request.user
        if not hasattr(user, "role") or user.role != "user":
            raise PermissionDenied("Only normal users can submit applications.")
        serializer.save(user=user)

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        user = request.user
        if instance.user != user:
            raise PermissionDenied("You can only update your own applications.")
        if instance.status != "pending":
            raise PermissionDenied("Cannot update application after it has been reviewed.")
        return super().update(request, *args, **kwargs)

    def partial_update(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        user = request.user
        if hasattr(user, "role") and user.role == "admin" or instance.user == user:
            return super().destroy(request, *args, **kwargs)
        raise PermissionDenied("You cannot delete this application.")

    @action(detail=True, methods=['post'], url_path='review')
    def review_application(self, request, pk=None):
        user = request.user
        if not hasattr(user, "role") or user.role != "admin":
            raise PermissionDenied("Only admins can review applications.")

        application = self.get_object()

        if application.status != "pending":
            return Response(
                {"detail": "Cannot review application that is already reviewed."},
                status=status.HTTP_400_BAD_REQUEST
            )

        status_value = request.data.get("status")
        if status_value not in ["approved", "rejected"]:
            return Response(
                {"detail": "Invalid status. Must be 'approved' or 'rejected'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Update application status
        application.status = status_value
        application.reviewed_by = user
        application.reviewed_at = timezone.now()
        application.save(update_fields=["status", "reviewed_by", "reviewed_at"])

        # Send email notification
        try:
            send_application_status_email(application.user, application, status_value)
        except Exception as e:
            print(f"Failed to send email: {e}")

        # Automatically approve pending domain/subdomain
        if status_value == "approved":
            if application.domain and application.domain.status == "pending":
                application.domain.status = "approved"
                application.domain.save(update_fields=["status"])

            if application.sub_domain and application.sub_domain.status == "pending":
                application.sub_domain.status = "approved"
                application.sub_domain.save(update_fields=["status"])

        # If approved, create a new Consultant
        consultant_data = None
        if status_value == "approved" and not Consultant.objects.filter(user=application.user).exists():
            consultant = Consultant.objects.create(
                user=application.user,
                location=application.location,
                description=application.description,
                years_experience=application.years_experience,
                domain=application.domain,
                sub_domain=application.sub_domain,
                cost=application.cost,
                validated=True,
                validated_by=user,
                validated_at=timezone.now(),
            )

            # Safe copy of photo
            if application.photo:
                try:
                    print(f"[DEBUG] Application has photo: {bool(application.photo)}")
                    original_file = application.photo.file_path
                    print(f"[DEBUG] Photo file path: {getattr(original_file, 'name', None)}")

                    # Use FileField open or fallback to filesystem path
                    if hasattr(original_file, 'open'):
                        f = original_file.open('rb')
                    else:
                        f = open(original_file.path, 'rb')

                    new_photo = Resource.objects.create(
                        file_path=File(f, name=os.path.basename(original_file.name)),
                        file_meta_data=application.photo.file_meta_data.copy() if application.photo.file_meta_data else None,
                        relation_type=ContentType.objects.get_for_model(Consultant),
                        relation_id=consultant.id,
                    )
                    consultant.photo = new_photo
                    consultant.save(update_fields=['photo'])
                    print(f"[DEBUG] Consultant photo created: ID={new_photo.id}, URL={new_photo.file_path.url}")
                except Exception as e:
                    print(f"[DEBUG] Failed to copy photo: {e}")

            # Update user role to consultant
            application.user.role = "consultant"
            application.user.save(update_fields=['role'])

            consultant_data = {
                "id": consultant.id,
                "user": consultant.user.email
            }

        print(f"[DEBUG] Consultant created: ID={consultant.id}, user={consultant.user.email}")

        response_data = {
            "id": application.id,
            "status": application.status,
            "reviewed_by": application.reviewed_by.email,
            "reviewed_at": application.reviewed_at,
            "consultant_created": consultant_data
        }

        return Response(response_data, status=status.HTTP_200_OK)


def send_application_status_email(user, application, status_value):
    subject = "Better Consult - Consultant Application Update"
    from_email = f"Better Consult <{getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@betterconsult.app')}>"

    if status_value == "approved":
        status_text = "approved"
        main_message = f"Congratulations {user.first_name or user.email}, your consultant application has been approved!"
        extra_message = "You can now log in as a consultant and start offering your services on Better Consult."
    else:
        status_text = "rejected"
        main_message = f"Hello {user.first_name or user.email}, unfortunately your consultant application has been rejected."
        extra_message = "You may review your application and submit again if you wish."

    text_content = (
        f"Your consultant application has been {status_text}.\n\n"
        f"{main_message}\n\n{extra_message}\n\n— Better Consult"
    )

    html_content = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; background:#f6f9fc; padding:24px;">
        <div style="max-width:520px; margin:0 auto; background:#ffffff; border-radius:8px; box-shadow:0 2px 8px rgba(16,24,40,0.05);">
            <div style="padding:20px 24px; border-bottom:1px solid #eef2f7;">
                <h2 style="margin:0; color:#0f172a; font-weight:700; font-size:18px;">Better Consult</h2>
            </div>
            <div style="padding:24px; color:#0f172a;">
                <p style="margin:0 0 12px; font-size:16px;">{main_message}</p>
                <div style="margin:20px 0; padding:12px 16px; background:#0ea5e9; color:#ffffff; font-weight:700; font-size:18px; border-radius:6px; display:inline-block;">
                    Status: {status_text}
                </div>
                <p style="margin:16px 0 0; color:#475569; font-size:14px;">{extra_message}</p>
                <p style="margin:8px 0 0; color:#64748b; font-size:12px;">If you have questions, please contact support.</p>
            </div>
            <div style="padding:16px 24px; border-top:1px solid #eef2f7; color:#94a3b8; font-size:12px;">
                © Better Consult. All rights reserved.
            </div>
        </div>
    </div>
    """

    message = EmailMultiAlternatives(subject, text_content, from_email, [user.email])
    message.attach_alternative(html_content, "text/html")
    message.send(fail_silently=False)
