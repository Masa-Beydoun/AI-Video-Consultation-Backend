# consulting/permissions.py
from rest_framework.permissions import BasePermission, SAFE_METHODS
from rest_framework import permissions

class IsAdminOrReadOnly(BasePermission):
    """
    Allow read-only requests for everyone authenticated,
    but only allow write requests for admins.
    """
    def has_permission(self, request, view):
        # SAFE_METHODS = ('GET', 'HEAD', 'OPTIONS')
        if request.method in SAFE_METHODS:
            return True
        
        # For write methods (POST, PUT, PATCH, DELETE) — must be admin
        return (
            request.user.is_authenticated
            and request.user.role == 'admin'
        )
class IsConsultant(permissions.BasePermission):
    """
    Only users with role 'consultant' can access.
    """

    def has_permission(self, request, view):
        user = request.user
        return user.is_authenticated and getattr(user, "role", None) == "consultant"