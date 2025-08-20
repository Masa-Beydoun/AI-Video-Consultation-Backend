from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.conf import settings
import random
import string
from datetime import timedelta

from .serializers import (
    UserRegistrationSerializer,
    UserLoginSerializer,
    CustomTokenObtainPairSerializer,
    OTPRequestSerializer,
    OTPVerifySerializer,
)
from django.contrib.auth import get_user_model


User = get_user_model()

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

# Helper function to generate JWT tokens for a user
def generate_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'token': str(refresh.access_token),
        'user': {
            'id': user.id,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'phone_number': user.phone_number,
            'role': user.role,
            'gender': user.gender,
            'is_active': user.is_active,
            # 'date_joined': user.date_joined,
            # 'last_login': user.last_login
        }
    }


# set up the email form
def send_better_consult_otp_email(recipient_email: str, otp: str, *, purpose: str = "Email verification") -> None:
    subject = f"Better Consult {purpose} code"
    from_email = f"Better Consult <{getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@betterconsult.app')}>"
    text_content = (
        f"Your Better Consult {purpose.lower()} code is: {otp}\n\n"
        "This code expires in 10 minutes. If you did not request this, you can safely ignore this email.\n\n"
        "— Better Consult"
    )
    html_content = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; background:#f6f9fc; padding:24px;">
      <div style="max-width:520px; margin:0 auto; background:#ffffff; border-radius:8px; box-shadow:0 2px 8px rgba(16,24,40,0.05);">
        <div style="padding:20px 24px; border-bottom:1px solid #eef2f7;">
          <h2 style="margin:0; color:#0f172a; font-weight:700; font-size:18px;">Better Consult</h2>
        </div>
        <div style="padding:24px; color:#0f172a;">
          <p style="margin:0 0 12px;">Use the code below to complete your {purpose.lower()}.</p>
          <div style="display:inline-block; padding:12px 16px; background:#0ea5e9; color:#ffffff; font-weight:700; letter-spacing:4px; font-size:20px; border-radius:6px;">
            {otp}
          </div>
          <p style="margin:16px 0 0; color:#475569;">This code expires in <strong>10 minutes</strong>.</p>
          <p style="margin:8px 0 0; color:#64748b; font-size:13px;">If you did not request this, please ignore this email.</p>
        </div>
        <div style="padding:16px 24px; border-top:1px solid #eef2f7; color:#94a3b8; font-size:12px;">
          © {timedelta(days=0).max.__class__.__name__ and ''}Better Consult. All rights reserved.
        </div>
      </div>
    </div>
    """
    message = EmailMultiAlternatives(subject=subject, body=text_content, from_email=from_email, to=[recipient_email])
    message.attach_alternative(html_content, "text/html")
    message.send(fail_silently=False)


# register 
@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        tokens = generate_tokens_for_user(user)
        return Response({
            'message': 'User registered successfully',
            'user_id': user.id,
            **tokens
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# login 
@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    serializer = UserLoginSerializer(data=request.data)
    if serializer.is_valid():
        email = serializer.validated_data['email']
        password = serializer.validated_data['password']
        user = authenticate(email=email, password=password)
        if user:
            if not user.is_active:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
            tokens = generate_tokens_for_user(user)
            return Response(tokens)
        else:
            return Response({'error': 'Invalid login information'}, status=status.HTTP_401_UNAUTHORIZED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# lgoout
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_user(request):
    try:
        return Response({'message': 'Logged out successfully'})
    except Exception as e:
        return Response({'error': f'Logout error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# sending OTP
@api_view(['POST'])
@permission_classes([AllowAny])
def send_otp(request):
    serializer = OTPRequestSerializer(data=request.data)
    if serializer.is_valid():
        email = serializer.validated_data['email']
        otp = ''.join(random.choices(string.digits, k=6))
        # Store OTP in cache for 10 minutes
        cache_key = f'otp_{email}'
        cache.set(cache_key, otp, 600) 
        # Send OTP via email
        try:
            send_better_consult_otp_email(email, otp, purpose="Email verification")
            return Response({
                'message': 'OTP sent successfully , check your email '
            })
        except Exception as e:
            # If email fails, return error
            return Response({
                'error': 'Failed to send OTP email',
                'email_error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# OTP verification
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp(request):
    serializer = OTPVerifySerializer(data=request.data)
    if serializer.is_valid():
        email = serializer.validated_data['email']
        otp = serializer.validated_data['otp']
        # Check OTP from cache
        cache_key = f'otp_{email}'
        stored_otp = cache.get(cache_key)
        if stored_otp and stored_otp == otp:
            # OTP is valid
            cache.delete(cache_key)  
            try:
                user = User.objects.get(email=email)
                # TODO: Add email_verified field to User model
                # user.email_verified = True
                # user.save()
                
                # Generate tokens for verified user
                tokens = generate_tokens_for_user(user)
                
                return Response({
                    'message': 'OTP verified successfully',
                })
            except User.DoesNotExist:
                return Response({
                    'error': 'User not found'
                }, status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({
                'error': 'Invalid OTP'
            }, status=status.HTTP_400_BAD_REQUEST)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def send_password_reset_otp(request):
    serializer = OTPRequestSerializer(data=request.data)
    if serializer.is_valid():
        email = serializer.validated_data['email']
        # Generate OTP
        otp = ''.join(random.choices(string.digits, k=6))
        cache_key = f'password_reset_otp_{email}'
        # Check if user exists 
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        try:
            user_exists = User.objects.filter(email=email).exists()
            if user_exists:
                cache.set(cache_key, otp, 600) 
                try:
                    send_better_consult_otp_email(
                        email, otp, purpose="Password Reset"
                    )
                except Exception as e:
                    pass
        except Exception:
            pass 
        return Response({
            'message': 'If you have an account on our app, you will receive an OTP shortly.'
        })

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



# reset password
@api_view(['POST'])
@permission_classes([AllowAny])
def confirm_password_reset(request):
    email = request.data.get('email')
    new_password = request.data.get('new_password')

    if not email:
        return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

    if not new_password:
        return Response({'error': 'New password is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = User.objects.get(email=email)
        user.set_password(new_password)
        user.save()

        # Generate new tokens for the user
        tokens = generate_tokens_for_user(user)

        return Response({
            'message': 'Password reset successfully',
            **tokens
        }, status=status.HTTP_200_OK)

    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_profile(request):
    user = request.user  # the logged-in user
    return Response({
        'id': user.id,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'phone_number': user.phone_number,
        'role': user.role,
        'gender': user.gender,
        'is_active': user.is_active,
    }, status=status.HTTP_200_OK)
