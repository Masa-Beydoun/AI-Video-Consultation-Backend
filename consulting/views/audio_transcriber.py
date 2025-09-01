import tempfile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

# from consulting.utils.audio_transcriber import transcribe_audio
from consulting.utils.audio_transcriber import transcribe_audio_file



class TranscriptionView(APIView):
    permission_classes = [IsAuthenticated] 
    
    def post(self, request):
        audio_file = request.FILES.get("file")
        if not audio_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            transcript = transcribe_audio_file(tmp_path)
            return Response({"transcript": transcript}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
