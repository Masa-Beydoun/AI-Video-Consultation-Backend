from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    register_user,
    login_user,
    send_otp,
    verify_otp,
    confirm_password_reset,
    logout_user,
    CustomTokenObtainPairView
)

urlpatterns = [

    # Authentication endpoints
    path('register/', register_user, name='register'),
    path('login/', login_user, name='login'),
    path('logout/', logout_user, name='logout'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),


    # OTP endpoints
    path('send-otp/', send_otp, name='send_otp'),
    path('verify-otp/', verify_otp, name='verify_otp'),

    # Password reset endpoints
    path('confirm-password-reset/', confirm_password_reset, name='confirm_password_reset'),
    
    
] 