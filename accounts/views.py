from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CustomUser, VerificationCode, AuthToken
from .serializers import RegisterSerializer
import random
from django.core.mail import send_mail

class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            code = str(random.randint(1000, 9999))
            VerificationCode.objects.create(user=user, code=code)
            # send email here
            send_mail("Your verification code", f"Code: {code}", "noreply@yourapp.com", [user.email])
            return Response({"detail": "Verification code sent."}, status=201)
        return Response(serializer.errors, status=400)

class VerifyCodeView(APIView):
    def post(self, request):
        email = request.data.get("email")
        code = request.data.get("code")
        user = CustomUser.objects.filter(email=email).first()
        if not user:
            return Response({"error": "User not found"}, status=404)

        verification = VerificationCode.objects.filter(user=user, code=code).first()
        if verification:
            VerificationCode.objects.filter(user=user).delete()
            token = AuthToken.objects.create(user=user)
            return Response({
                "token": token.token,
                "user": RegisterSerializer(user).data
            }, status=200)
        return Response({"error": "Invalid code"}, status=400)
    
from django.contrib.auth import authenticate

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(email=email, password=password)
        if user:
            token = AuthToken.objects.create(user=user)
            return Response({
                "token": token.token,
                "user": RegisterSerializer(user).data
            }, status=200)
        return Response({"error": "Invalid credentials"}, status=400)

class LogoutView(APIView):
    def post(self, request):
        token_str = request.headers.get('Authorization')
        if token_str:
            AuthToken.objects.filter(token=token_str).delete()
            return Response({"detail": "Logged out"}, status=200)
        return Response({"error": "No token provided"}, status=400)
