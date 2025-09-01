from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import Message, Chat
from consulting.models.consultant import Consultant
from .serializers import MessageSerializer, ChatSerializer, ConsultantSerializer
from rest_framework.generics import ListAPIView, DestroyAPIView
from django.core.files.storage import default_storage

# Ask question
class MessageCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user = request.user  # from auth token

        consultant_id = request.data.get("consultant_id")
        text = request.data.get("text")

        if not consultant_id or not text:
            return Response(
                {"error": "consultant_id and text are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate consultant
        try:
            consultant = Consultant.objects.get(id=consultant_id)
        except Consultant.DoesNotExist:
            return Response({"error": "Consultant not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get or create chat
        chat, created = Chat.objects.get_or_create(
            user=user,
            consultant=consultant,
            defaults={"title": f"Chat with {consultant}"}
        )

        # Save user message
        user_message = Message.objects.create(
            chat=chat,
            sender="U",
            text=text
        )

        # ---- Generate consultant reply ----
        reply_text = self.generate_reply(user_message.text)

        reply = Message.objects.create(
            chat=chat,
            sender="C",
            text=reply_text
        )

        chat.modified_at = reply.sent_at
        chat.save(update_fields=["modified_at"])

        return Response({
            "chat_id": chat.id,
            "user_message": MessageSerializer(user_message).data,
            "consultant_message": MessageSerializer(reply).data
        }, status=status.HTTP_201_CREATED)

    def generate_reply(self, user_text):
        # AI system
        return f"Auto-reply to: {user_text}"


# All chats
class ChatListView(ListAPIView):
    serializer_class = ChatSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user).order_by('-modified_at')
    

# All messages in chat
class ConsultantChatMessagesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        consultant_id = request.data.get("consultant_id")
        if not consultant_id:
            return Response({"error": "consultant_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            chat = Chat.objects.get(user=request.user, consultant_id=consultant_id)
        except Chat.DoesNotExist:
            return Response({"error": "No chat found with this consultant."}, status=status.HTTP_404_NOT_FOUND)

        messages = Message.objects.filter(chat=chat).order_by("-sent_at")
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


#Delete chat
class ChatDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, *args, **kwargs):
        chat_id = request.data.get("chat_id")
        if not chat_id:
            return Response(
                {"error": "chat_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            chat = Chat.objects.get(id=chat_id, user=request.user)
        except Chat.DoesNotExist:
            return Response(
                {"error": "Chat not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        chat.delete()
        return Response(
            {"message": "Chat deleted successfully"},
            status=status.HTTP_204_NO_CONTENT
        )

# Question without specified consultant
class QuestionConsultantsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        question = request.data.get("question")
        if not question:
            return Response({"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST)

        consultants = Consultant.objects.all()

        if not consultants.exists():
            return Response(
                {"message": f"Nobody answered this question: '{question}'"},
                status=status.HTTP_200_OK
            )

        serializer = ConsultantSerializer(consultants, many=True)
        return Response({
            "question": question,
            "consultants": serializer.data
        }, status=status.HTTP_200_OK)


ALLOWED_AUDIO_TYPES = [
    "audio/mpeg",      # .mp3
    "audio/wav",       # .wav
    "audio/x-wav",
    "audio/webm",      # .webm
    "audio/ogg",       # .ogg
    "audio/flac",      # .flac
    "audio/mp4",       # .m4a 
    "video/mp4",       # .mp4 
]

# Voice question
class VoiceToTextView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        audio_file = request.FILES.get("voice")
        if not audio_file:
            return Response({"error": "No voice file provided"}, status=status.HTTP_400_BAD_REQUEST)

        if audio_file.content_type not in ALLOWED_AUDIO_TYPES:
            return Response(
                {"error": f"Invalid file type: {audio_file.content_type}. Please upload a valid audio file."},
                status=status.HTTP_400_BAD_REQUEST
            )

        text = self.voice_transcription(audio_file)

        return Response({"transcription": text}, status=status.HTTP_200_OK)
    
    def voice_transcription(self, audio_file):
        return "This is a dummy transcription of the voice."


