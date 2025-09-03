from langdetect import detect
from gtts import gTTS
import pygame
import os
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tts_multilang(text):
    try:
        # Detect language
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        
        # Create a mapping of detected languages to gTTS language codes
        language_map = {
            'en': 'en',    # English
            'ar': 'ar',    # Arabic
            'es': 'es',     # Spanish
            'fr': 'fr',     # French
            'de': 'de',     # German
            'it': 'it',     # Italian
            'pt': 'pt',    # Portuguese
            'ru': 'ru',     # Russian
            'zh': 'zh-CN', # Chinese
            'ja': 'ja',     # Japanese
            'ko': 'ko'      # Korean
        }
        
        # Default to English if language not in map
        tts_lang = language_map.get(lang, 'en')
        
        # Create in-memory file
        mp3_fp = BytesIO()
        
        # Generate speech
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Play audio
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_fp)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        logger.error(f"Error processing text '{text}': {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    texts = [
        "Hello,amna how are you?",  # English
        "مرحبا امنه كيف حالك؟",       # Arabic
        "Hola,amna ¿cómo estás?",   # Spanish
        "Bonjour,amna comment ça va?" # French
    ]
    
    for txt in texts:
        try:
            tts_multilang(txt)
        except Exception as e:
            logger.error(f"Skipping text due to error: {str(e)}")
            continue
    
    # Clean up pygame
    pygame.quit()

    ############doneeeeeeee