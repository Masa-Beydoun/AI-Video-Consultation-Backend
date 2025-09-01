from django.urls import path
from .views import MessageCreateView
from .views import ChatListView
from .views import ConsultantChatMessagesView
from .views import ChatDeleteView

urlpatterns = [
    path('ask/', MessageCreateView.as_view(), name='ask-question'),
    path('get_user_chats/', ChatListView.as_view(), name='chat-list'),
    path('messages/', ConsultantChatMessagesView.as_view(), name='chat-messages'),
    path('delete/', ChatDeleteView.as_view(), name='chat-delete'),
]