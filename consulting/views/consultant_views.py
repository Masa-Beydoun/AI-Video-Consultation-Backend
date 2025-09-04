from rest_framework import viewsets, permissions, status, filters
from rest_framework.exceptions import PermissionDenied, NotFound
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.permissions import IsAuthenticated
from consulting.models.consultant import Consultant
from consulting.serializers.consultant_serializer import ConsultantSerializer

class ConsultantViewSet(viewsets.ModelViewSet):
    queryset = Consultant.objects.all()
    serializer_class = ConsultantSerializer
    permission_classes = [permissions.IsAuthenticated]  # all endpoints require login

    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['domain', 'sub_domain']

    # 🔍 traverse into user relation
    search_fields = [
        'user__first_name', 'user__last_name', 'user__phone_number',   
    ]

    def get_queryset(self):
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
    
    @action(detail=False, methods=['get'], url_path=r'by-domain-subdomain/(?P<domain_id>\d+)/(?P<subdomain_id>\d+)')
    def get_by_domain_and_subdomain(self, request, domain_id=None, subdomain_id=None):
        """
        Get all consultants by domain id and subdomain id
        """
        consultants = Consultant.objects.filter(domain_id=domain_id, sub_domain_id=subdomain_id)
        if not consultants.exists():
            return Response(
                {"detail": "No consultants found for this domain and subdomain."},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = self.get_serializer(consultants, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['get'], url_path=r'by-domain/(?P<domain_id>\d+)')
    def get_by_domain(self, request, domain_id=None):
        """
        Get all consultants in a given domain by domain id
        """
        consultants = Consultant.objects.filter(domain_id=domain_id)
        if not consultants.exists():
            return Response(
                {"detail": "No consultants found for this domain."},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = self.get_serializer(consultants, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['get'], url_path='top-rated')
    def top_rated(self, request):
        """
        Return the 10 highest-rated consultants
        GET /api/consultants/top-rated/
        """
        consultants = Consultant.objects.order_by('-rating', '-review_count')[:10]
        serializer = self.get_serializer(consultants, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], url_path='top-20-rated')
    def top_20_rated(self, request):
        """
        Return the 20 highest-rated consultants
        GET /api/consultants/top-20-rated/
        """
        consultants = Consultant.objects.order_by('-rating', '-review_count')[:20]
        serializer = self.get_serializer(consultants, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


    @action(detail=False, methods=['get', 'patch'], url_path='me', permission_classes=[IsAuthenticated])
    def me(self, request):
        """
        GET: return the profile of the logged-in consultant.
        PATCH: update the profile of the logged-in consultant.
        """
        try:
            consultant = Consultant.objects.get(user=request.user)
        except Consultant.DoesNotExist:
            return Response(
                {"detail": "You do not have a consultant profile."},
                status=status.HTTP_403_FORBIDDEN
            )

        if request.method == 'GET':
            serializer = self.get_serializer(consultant)
            return Response(serializer.data, status=status.HTTP_200_OK)

        elif request.method == 'PATCH':
            serializer = self.get_serializer(consultant, data=request.data, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)