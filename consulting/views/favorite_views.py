from rest_framework import serializers 
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from consulting.models.favorite import Favorite
from consulting.serializers.favorite_serializer import FavoriteSerializer

class FavoriteView(APIView):
    permission_classes = [IsAuthenticated]  # ensure user is logged in

    def post(self, request):
        serializer = FavoriteSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        serializer = FavoriteSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            try:
                serializer.remove()
                return Response({"detail": "Favorite removed."}, status=status.HTTP_204_NO_CONTENT)
            except serializers.ValidationError as e:
                return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
