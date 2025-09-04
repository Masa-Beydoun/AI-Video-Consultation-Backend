# consulting/utils/video_checks.py
import os
import tempfile
import numpy as np
from django.conf import settings

# audio
from pydub import AudioSegment, silence
import librosa
from django.db.models import Avg

# video / frames
import cv2

# face / identity
# deepface, face_recognition, mediapipe are optional and heavy; imported lazily

def _safe_load_audio(path):
    try:
        audio = AudioSegment.from_file(path)
        return audio
    except Exception as e:
        raise RuntimeError(f"Could not load audio: {e}")

def check_audio_loudness(path):
    try:
        audio = _safe_load_audio(path)
        loudness = audio.dBFS
        if loudness < -30:
            return {"status": "Too Quiet", "value_dbfs": round(loudness, 2)}
        elif loudness > -5:
            return {"status": "Too Loud (Might Clip)", "value_dbfs": round(loudness, 2)}
        else:
            return {"status": "Good Volume", "value_dbfs": round(loudness, 2)}
    except Exception as e:
        return {"error": str(e)}

def calculate_snr(path, noise_duration_ms=1000):
    try:
        audio = _safe_load_audio(path)
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if len(samples) == 0:
            return {"error": "Empty audio"}
        sr = audio.frame_rate
        n_noise = int(sr * noise_duration_ms / 1000)
        noise_samples = samples[:n_noise] if len(samples) > n_noise else samples
        signal_power = np.mean(samples**2)
        noise_power = np.mean(noise_samples**2) if len(noise_samples) > 0 else 0
        if noise_power == 0:
            return {"snr_db": None, "note": "No detectable noise (noise power=0)"}
        snr = 10 * np.log10(signal_power / noise_power)
        return {"snr_db": round(snr, 2)}
    except Exception as e:
        return {"error": str(e)}

def detect_silence_periods(path, silence_thresh=-40, min_silence_len=2000):
    try:
        audio = _safe_load_audio(path)
        audio = audio.set_channels(1)
        silent_ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        silent_durations = [(end - start) / 1000.0 for start, end in silent_ranges]
        total_silence = sum(silent_durations)
        return {
            "silent_periods": len(silent_ranges),
            "total_silence_seconds": round(total_silence, 2),
            "status": (
                "Too much silence" if total_silence > 5
                else "Frequent silent pauses" if len(silent_ranges) > 3
                else "Silence level is okay"
            )
        }
    except Exception as e:
        return {"error": str(e)}

def detect_audio_issues(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) >= clipping_threshold)
        clip_ratio = float(clipped_samples) / max(1, len(y))
        flatness = librosa.feature.spectral_flatness(y=y + 1e-12)
        avg_flatness = float(np.mean(flatness))
        messages = []
        if clip_ratio > 0.01:
            messages.append("Clipping detected")
        if avg_flatness > 0.4:
            messages.append("Echo or noise-like signal detected")
        return {"clip_ratio": clip_ratio, "avg_spectral_flatness": round(avg_flatness, 4),
                "status": "Audio is clean" if not messages else "; ".join(messages)}
    except Exception as e:
        return {"error": str(e)}

def detect_black_screen(path, sample_rate=30, black_threshold=0.6):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        black_frames = 0
        total_sampled = 0
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
                if brightness < 10:  # threshold for black
                    black_frames += 1
                total_sampled += 1
            frame_index += 1
        cap.release()
        if total_sampled == 0:
            return {"error": "Video empty or unreadable"}
        black_ratio = black_frames / total_sampled
        status = ("Mostly black screen" if black_ratio > black_threshold
                  else "Some black frames" if black_ratio > 0.1
                  else "Video is fine")
        return {"black_frame_ratio": round(black_ratio, 3), "status": status}
    except Exception as e:
        return {"error": str(e)}

def detect_blurriness(path, sample_rate=30, blurry_threshold=100.0):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        blurry_frames = 0
        total_sampled = 0
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < blurry_threshold:
                    blurry_frames += 1
                total_sampled += 1
            frame_index += 1
        cap.release()
        if total_sampled == 0:
            return {"error": "Video empty or unreadable"}
        blur_ratio = blurry_frames / total_sampled
        status = ("Video is too blurry" if blur_ratio > 0.6
                  else "Some blurriness" if blur_ratio > 0.2
                  else "Video is sharp enough")
        return {"blurry_frame_ratio": round(blur_ratio, 3), "status": status}
    except Exception as e:
        return {"error": str(e)}

def check_resolution(path, min_width=640, min_height=480):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        status = "Resolution is good" if width >= min_width and height >= min_height else f"Resolution too low: {width}x{height}"
        return {"resolution": f"{width}x{height}", "status": status}
    except Exception as e:
        return {"error": str(e)}

def check_frame_rate(path, min_fps=24):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        status = ("Frame rate is good" if fps >= min_fps else "Frame rate is low")
        return {"fps": round(float(fps), 2), "status": status}
    except Exception as e:
        return {"error": str(e)}

# DeepFace / identity checks (optional — heavy)
def _extract_first_face(video_path, output_image_path="temp_face.jpg", max_frames=300):
    try:
        import cv2
        from deepface.commons import functions
        cap = cv2.VideoCapture(video_path)
        face_saved = False
        index = 0
        while cap.isOpened() and index < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                face_img = functions.detectFace(frame, detector_backend='ssd')
                if face_img is not None:
                    cv2.imwrite(output_image_path, face_img)
                    face_saved = True
                    break
            except Exception:
                pass
            index += 1
        cap.release()
        return (face_saved, output_image_path if face_saved else None)
    except Exception as e:
        return (False, str(e))

def verify_identity(reference_image_path, video_path):
    try:
        from deepface import DeepFace
    except Exception as e:
        return {"error": f"DeepFace not available: {e}"}
    ok, face_path = _extract_first_face(video_path)
    if not ok:
        return {"status": "could_not_extract_face"}
    try:
        result = DeepFace.verify(img1_path=reference_image_path, img2_path=face_path, model_name="Facenet", enforce_detection=False)
        verified = result.get("verified", False)
        distance = result.get("distance")
        threshold = result.get("threshold")
        # remove extracted face file
        try:
            os.remove(face_path)
        except Exception:
            pass
        return {"verified": bool(verified), "distance": distance, "threshold": threshold}
    except Exception as e:
        return {"error": str(e)}

def detect_face_consistency(video_path, sample_rate=30, min_detection_ratio=0.7):
    try:
        import face_recognition
    except Exception:
        return {"error": "face_recognition not installed"}
    cap = cv2.VideoCapture(video_path)
    total_sampled = 0
    face_detected = 0
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_rate == 0:
            total_sampled += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb, model="hog")
            if faces:
                face_detected += 1
        frame_index += 1
    cap.release()
    if total_sampled == 0:
        return {"error": "video unreadable"}
    ratio = face_detected / total_sampled
    return {"detection_ratio": round(ratio, 3),
            "status": ("Face detected consistently" if ratio >= min_detection_ratio else "Face not detected reliably")}

def run_all_checks(video_path, reference_image_path=None):
    results = {}
    # audio checks
    results["audio_loudness"] = check_audio_loudness(video_path)
    results["snr"] = calculate_snr(video_path)
    results["audio_issues"] = detect_audio_issues(video_path)
    results["silence"] = detect_silence_periods(video_path)
    # video checks
    results["black_screen"] = detect_black_screen(video_path)
    results["blurriness"] = detect_blurriness(video_path)
    results["resolution"] = check_resolution(video_path)
    results["frame_rate"] = check_frame_rate(video_path)
    results["face_consistency"] = detect_face_consistency(video_path)
    # identity (optional)
    if reference_image_path:
        results["identity_verification"] = verify_identity(reference_image_path, video_path)
    return results

def run_audio_checks(audio_path):
    """
    Run all available audio checks on the given file.
    Returns a dictionary with results from each check.
    """
    results = {}
    
    # Loudness check
    results["loudness"] = check_audio_loudness(audio_path)
    
    # Signal-to-noise ratio
    results["snr"] = calculate_snr(audio_path)
    
    # Silence detection
    results["silence"] = detect_silence_periods(audio_path)
    
    # Other audio issues (clipping, flatness, etc.)
    results["audio_issues"] = detect_audio_issues(audio_path)
    
    return results

