# # views/auth_views.py
# import random
# import json
# from django.core.mail import send_mail
# from rest_framework.response import Response
# from consulting.serializers.user_serializer import UserRegisterSerializer
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.utils import timezone
# from ..models import User, EmailVerificationCode  
# from rest_framework.authentication import TokenAuthentication
# from rest_framework.decorators import api_view, authentication_classes, permission_classes
# from rest_framework.permissions import IsAuthenticated
# from django.contrib.auth import authenticate,get_user_model
# from rest_framework.authtoken.models import Token
# from rest_framework_simplejwt.views import TokenObtainPairView
# from rest_framework_simplejwt.tokens import RefreshToken
# from rest_framework import status, permissions
# from rest_framework_simplejwt.serializers import TokenObtainPairSerializer



# @api_view(['POST'])
# def register(request):
#     serializer = UserRegisterSerializer(data=request.data)
#     if serializer.is_valid():
#         user = serializer.save()
        
#         # Generate 4-digit code
#         code = f"{random.randint(1000, 9999)}"
#         EmailVerificationCode.objects.create(user=user, code=code)

#         # Send email
#         send_mail(
#             subject="Your verification code",
#             message=f"Your code is: {code}",
#             from_email="masa.beydoun@hotmail.com",
#             recipient_list=[user.email],
#             fail_silently=False,
#         )

#         return Response({"message": "User created. Verification code sent to email."})
#     return Response(serializer.errors, status=400)



# @api_view(['POST'])
# def login_user(request):
#     data = request.data
#     email = data.get('email')
#     password = data.get('password')

#     user = authenticate(request, email=email, password=password)

#     if user is not None:
#         if not user.is_verified:
#             return JsonResponse({'error': 'Email not verified'}, status=403)

#         token, created = Token.objects.get_or_create(user=user)
#         user_data = {
#             'id': user.id,
#             'email': user.email,
#             'first_name': user.first_name,
#             'last_name': user.last_name,
#             'phone_number': user.phone_number,
#             'role': user.role,
#             'gender': user.gender,
#         }

#         return JsonResponse({'token': token.key, 'user': user_data})
#     else:
#         return JsonResponse({'error': 'Invalid credentials'}, status=401)

# @api_view(['POST'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([IsAuthenticated])
# def logout_user(request):
#     request.user.auth_token.delete()
#     return JsonResponse({'message': 'Logged out successfully'})


# @api_view(['POST'])
# def forgot_password(request):
#     email = request.data.get('email')
#     try:
#         user = User.objects.get(email=email)
#         code = f"{random.randint(1000, 9999)}"
#         EmailVerificationCode.objects.create(user=user, code=code)

#         send_mail(
#             subject="Password Reset Code",
#             message=f"Your reset code is: {code}",
#             from_email="masa.beydoun@hotmail.com",
#             recipient_list=[email],
#             fail_silently=False,
#         )

#         return Response({"message": "Reset code sent to email"})
#     except User.DoesNotExist:
#         return Response({"error": "User not found"}, status=404)


# @api_view(['POST'])
# def reset_password(request):
#     email = request.data.get('email')
#     code = request.data.get('code')
#     new_password = request.data.get('new_password')

#     try:
#         user = User.objects.get(email=email)
#         verif = EmailVerificationCode.objects.filter(user=user, code=code).last()
#         if verif and not verif.is_expired():
#             user.password = make_password(new_password)
#             user.save()
#             return Response({"message": "Password reset successful"})
#         else:
#             return Response({"error": "Invalid or expired code"}, status=400)
#     except User.DoesNotExist:
#         return Response({"error": "User not found"}, status=404)



# @csrf_exempt
# def verify_email(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

#     try:
#         data = json.loads(request.body)
#         email = data.get('email')
#         code = data.get('code')

#         if not email or not code:
#             return JsonResponse({'error': 'Email and code are required.'}, status=400)

#         try:
#             verification = EmailVerificationCode.objects.get(user__email=email, code=code)
#         except EmailVerificationCode.DoesNotExist:
#             return JsonResponse({'error': 'Invalid verification code.'}, status=400)

#         if verification.is_expired():
#             verification.delete()
#             return JsonResponse({'error': 'Verification code expired.'}, status=400)

#         try:
#             user = User.objects.get(email=email)
#             user.is_verified = True  # ⬅️ Make sure your User model has this field
#             user.save()
#         except User.DoesNotExist:
#             return JsonResponse({'error': 'User not found.'}, status=404)

#         verification.delete()  # Optional: clean up used code

#         return JsonResponse({'message': 'Email verified successfully.'}, status=200)

#     except json.JSONDecodeError:
#         return JsonResponse({'error': 'Invalid JSON.'}, status=400)



# User = get_user_model()

# # Custom serializer to include user info in login response
# class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
#     def validate(self, attrs):
#         data = super().validate(attrs)
#         user = self.user
#         data['user'] = {
#             "id": user.id,
#             "email": user.email,
#             "first_name": user.first_name,
#             "last_name": user.last_name,
#             "role": user.role,
#             "gender": user.gender,
#             "is_verified": user.is_verified,
#         }
#         return data

# # View for login
# class CustomLoginView(TokenObtainPairView):
#     serializer_class = CustomTokenObtainPairSerializer

# # View for logout (just blacklists the refresh token)
# class LogoutView(TokenObtainPairView):
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request):
#         try:
#             refresh_token = request.data["refresh"]
#             token = RefreshToken(refresh_token)
#             token.blacklist()
#             return Response({"message": "Successfully logged out"}, status=status.HTTP_205_RESET_CONTENT)
#         except Exception as e:
#             return Response({"error": "Invalid token or token already blacklisted."}, status=status.HTTP_400_BAD_REQUEST)
