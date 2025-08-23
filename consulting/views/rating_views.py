# from rest_framework import status
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import IsAuthenticated
# from django.db.models import Avg
# from consulting.models.userconsultation import UserConsultation
# from consulting.models.consultant import Consultant

# class RateConsultantView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request):
#         user = request.user
#         consultant_id = request.data.get('consultant_id')
#         rate = request.data.get('rate')

#         # Validate input
#         if consultant_id is None or rate is None:
#             return Response({"detail": "consultant_id and rate are required."}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             rate = int(rate)
#             if rate < 0 or rate > 5:
#                 raise ValueError()
#         except ValueError:
#             return Response({"detail": "Rate must be an integer between 0 and 5."}, status=status.HTTP_400_BAD_REQUEST)

#         # Make sure the consultant exists
#         try:
#             consultant = Consultant.objects.get(id=consultant_id)
#         except Consultant.DoesNotExist:
#             return Response({"detail": "Consultant not found."}, status=status.HTTP_404_NOT_FOUND)

#        # Save or update the user rating
#         user_rating, created = UserConsultation.objects.update_or_create(
#             user=request.user,
#             consultation=consultant,
#             defaults={'rate': rate}
#         )

#         # Recalculate average
#         average_rating = UserConsultation.objects.filter(consultation=consultant).aggregate(Avg('rate'))['rate__avg']

#         # Update consultant's average rate field
#         consultant.rate = average_rating
#         consultant.save()

