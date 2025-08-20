# consulting/views/review_view.py
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from consulting.models.review import Review
from consulting.serializers.review_serializer import ReviewSerializer
from consulting.models.consultant import Consultant

class ReviewViewSet(viewsets.ModelViewSet):
    queryset = Review.objects.all()
    serializer_class = ReviewSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        review = serializer.save(user=self.request.user)
        review.consultant.update_rating()  

