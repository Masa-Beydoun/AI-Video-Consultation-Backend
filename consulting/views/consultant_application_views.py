from rest_framework import viewsets, status, permissions
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, NotFound
from django.utils import timezone

from consulting.models.consultant_application import ConsultantApplication
from consulting.models.resource import Resource
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
        # Use default IsAuthenticated for all
        permissions_list = [permissions.IsAuthenticated()]
        return permissions_list

    def retrieve(self, request, *args, **kwargs):
        """Only admin or the owner can retrieve, include related resources"""
        instance = self.get_object()
        user = request.user

        # Permission check
        if not (hasattr(user, "role") and user.role == "admin") and instance.user != user:
            raise PermissionDenied("You do not have permission to view this application.")

        # Serialize the application
        serializer = self.get_serializer(instance)

        # Get all resources related to this application
        resources = Resource.objects.filter(
            relation_type__model='consultantapplication', 
            relation_id=instance.id
        )
        resources_serialized = ResourceSerializer(resources, many=True, context={'request': request}).data

        # Combine the application and resources
        data = serializer.data
        data['resources'] = resources_serialized

        return Response(data)

    def perform_create(self, serializer):
        """Only normal users can create"""
        user = self.request.user
        if not hasattr(user, "role") or user.role != "user":
            raise PermissionDenied("Only normal users can submit applications.")
        serializer.save(user=user)

    def update(self, request, *args, **kwargs):
        """Only the owner can update if not approved/rejected"""
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
        """Only owner or admin can delete"""
        instance = self.get_object()
        user = request.user
        if hasattr(user, "role") and user.role == "admin":
            return super().destroy(request, *args, **kwargs)
        elif instance.user == user:
            return super().destroy(request, *args, **kwargs)
        else:
            raise PermissionDenied("You cannot delete this application.")

    @action(detail=True, methods=['post'], url_path='review')
    def review_application(self, request, pk=None):
        """Only admin can approve/reject"""
        user = request.user
        if not hasattr(user, "role") or user.role != "admin":
            raise PermissionDenied("Only admins can review applications.")

        application = self.get_object()
        status_value = request.data.get("status")
        if status_value not in ["approved", "rejected"]:
            return Response(
                {"detail": "Invalid status. Must be 'approved' or 'rejected'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        application.status = status_value
        application.reviewed_by = user
        application.reviewed_at = timezone.now()
        application.save()

        # ✅ Automatically approve domain/subdomain if they are pending
        if status_value == "approved":
            if application.domain and application.domain.status == "pending":
                application.domain.status = "approved"
                application.domain.save(update_fields=["status"])

            if application.sub_domain and application.sub_domain.status == "pending":
                application.sub_domain.status = "approved"
                application.sub_domain.save(update_fields=["status"])

        # If approved, create a Consultant
        consultant_data = None
        if status_value == "approved":
            if not hasattr(application.user, "consultant_profile"):
                from consulting.models.consultant import Consultant  
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

                # Copy the photo if exists
                if application.photo:
                    from django.contrib.contenttypes.models import ContentType
                    new_photo = Resource.objects.create(
                        file=application.photo.file,
                        relation_type=ContentType.objects.get_for_model(Consultant),
                        relation_id=consultant.id,
                        file_meta_data=application.photo.file_meta_data,
                    )
                    consultant.photo = new_photo
                    consultant.save(update_fields=["photo"])

                # Update user role to 'consultant'
                application.user.role = "consultant"
                application.user.save(update_fields=['role'])

                consultant_data = {
                    "id": consultant.id,
                    "user": consultant.user.email
                }

        response_data = {
            "id": application.id,
            "status": application.status,
            "reviewed_by": application.reviewed_by.email,
            "reviewed_at": application.reviewed_at,
            "consultant_created": consultant_data
        }

        return Response(response_data, status=status.HTTP_200_OK)
