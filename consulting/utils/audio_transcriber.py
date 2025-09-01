# myapp/utils/audio_transcriber.py
import whisper

# Load model once globally
model = whisper.load_model('small')  # choose tiny, base, small, medium, large

def transcribe_audio_file(file_path: str) -> str:
    """
    Transcribe an audio file (mp3, wav, etc.) into text using Whisper.
    
    """
    result = model.transcribe(file_path, fp16=False)
    return result['text']
