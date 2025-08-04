# from django.urls import path
# from ..views import auth_views
# from rest_framework_simplejwt.views import TokenRefreshView
# from ..views.auth_views import CustomLoginView, LogoutView  # We'll define these

# urlpatterns = [
#     path('register/', auth_views.register, name='register'),
#     path('verify-email/', auth_views.verify_email, name='verify_email'),
#     path('login/', CustomLoginView.as_view(), name='login'),  # Custom JWT Login with user info
#     path('logout/', LogoutView.as_view(), name='logout'),     # Custom logout to invalidate token
#     path('forgot-password/', auth_views.forgot_password, name='forgot_password'),
#     path('reset-password/', auth_views.reset_password, name='reset_password'),
#     path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
# ]
