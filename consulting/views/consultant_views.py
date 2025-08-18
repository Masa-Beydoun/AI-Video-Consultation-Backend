from rest_framework import viewsets, permissions, status
from rest_framework.exceptions import PermissionDenied, NotFound
from rest_framework.decorators import action
from rest_framework.response import Response

from consulting.models.consultant import Consultant
from consulting.serializers.consultant_serializer import ConsultantSerializer

class ConsultantViewSet(viewsets.ModelViewSet):
    queryset = Consultant.objects.all()
    serializer_class = ConsultantSerializer
    permission_classes = [permissions.IsAuthenticated]  # all endpoints require login

    def get_queryset(self):
        # Everyone can see all consultants
        return Consultant.objects.all()

    def retrieve(self, request, *args, **kwargs):
        # Everyone can retrieve a consultant
        return super().retrieve(request, *args, **kwargs)

    @action(detail=False, methods=['patch'], url_path='me')
    def update_self(self, request):
        """
        Update the consultant linked to the authenticated user.
        PATCH /api/consultants/me/
        """
        try:
            consultant = Consultant.objects.get(user=request.user)
        except Consultant.DoesNotExist:
            return Response(
                {"detail": "You do not have a consultant profile."},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = self.get_serializer(consultant, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
