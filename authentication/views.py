from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth import get_user_model
from .serializers import (
    CustomTokenObtainPairSerializer,
    UserRegistrationSerializer,
    UserLoginSerializer
)

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
