from django.urls import path
from .views import MessageCreateView
from .views import ChatListView
from .views import ConsultantChatMessagesView
from .views import ChatDeleteView
from .views import QuestionConsultantsView
from .views import VoiceToTextView
from .views import WaitingQuestionView
from .views import QuestionDeleteView

urlpatterns = [
    path('ask/', MessageCreateView.as_view(), name='ask-question'),
    path('get_user_chats/', ChatListView.as_view(), name='chat-list'),
    path('messages/', ConsultantChatMessagesView.as_view(), name='chat-messages'),
    path('delete/', ChatDeleteView.as_view(), name='chat-delete'),
    path('question_consultants/', QuestionConsultantsView.as_view(), name='consultants-by-question'),
    path("voice_to_text/", VoiceToTextView.as_view(), name="voice-to-text"),
    path("waiting_questions_list/", WaitingQuestionView.as_view(), name="waiting-questions-list"),
    path("delete_question/", QuestionDeleteView.as_view(), name="delete-question"),
]