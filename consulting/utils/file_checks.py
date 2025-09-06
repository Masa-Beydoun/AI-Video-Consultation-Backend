# consulting/utils/video_checks.py
import numpy as np
import cv2
from pydub import AudioSegment, silence
import librosa
from deepface import DeepFace
import os
from moviepy import VideoFileClip
import tempfile



class FaceVerificationSystem:
    def __init__(self, model_name: str = "Facenet", similarity_threshold: float = 0.6):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.photo_embedding = None

    def preprocess_photo(self, photo_path: str):
        self.photo_embedding = DeepFace.represent(
            img_path=photo_path,
            model_name=self.model_name,
            detector_backend="retinaface",
            enforce_detection=True
        )[0]["embedding"]

    def cosine_similarity(self, vec1, vec2):
        v1, v2 = np.array(vec1), np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def normalize_frame(self, frame):
        """Normalize frame to 8-bit RGB format for DeepFace compatibility"""
        # Ensure frame is not empty
        if frame is None or frame.size == 0:
            return None
            
        # Convert to uint8 if not already
        if frame.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure it's 3-channel BGR (OpenCV format)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3:
            if frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 1:  # Single channel
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        return frame

    def best_orientation_similarity(self, frame: np.ndarray) -> float:
        # Normalize frame first
        frame = self.normalize_frame(frame)
        if frame is None:
            return -1
            
        rotations = [
            frame,
            cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(frame, cv2.ROTATE_180),
            cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]
        
        for i, rotated in enumerate(rotations):
            try:
                # Save frame to temporary file for DeepFace
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    cv2.imwrite(temp_path, rotated)
                
                try:
                    result = DeepFace.represent(
                        img_path=temp_path,
                        model_name=self.model_name,
                        detector_backend="retinaface",
                        enforce_detection=True
                    )
                    emb = result[0]["embedding"]
                    sim = self.cosine_similarity(self.photo_embedding, emb)
                    if sim >= self.similarity_threshold or i == len(rotations) - 1:
                        return sim
                finally:
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
            except Exception as e:
                print(f"[WARN] DeepFace failed on {i*90}°: {e}")
                continue
        return -1

    def verify_identity(self, photo_path: str, video_path: str, max_frames=10):
        if self.photo_embedding is None:
            self.preprocess_photo(photo_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        sims, matches, processed = [], 0, 0
        idx = 0

        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret: 
                break
            if idx % frame_interval == 0:
                sim = self.best_orientation_similarity(frame)
                if sim >= 0:
                    sims.append(sim)
                    if sim >= self.similarity_threshold:
                        matches += 1
                processed += 1
            idx += 1
        cap.release()

        if not sims:
            return {"status": "no_faces_detected"}

        return {
            "verified": (np.mean(sims) >= self.similarity_threshold and matches / len(sims) >= 0.3),
            "confidence": float((np.mean(sims) + np.max(sims)) / 2),
            "frames_processed": processed,
            "matches": matches,
            "max_similarity": float(np.max(sims)),
            "avg_similarity": float(np.mean(sims)),
        }

def _safe_load_audio(path):
    try:
        audio = AudioSegment.from_file(path)
        return audio
    except Exception as e:
        raise RuntimeError(f"Could not load audio: {e}")

# ---------------- Audio Checks ----------------
def check_audio_loudness(path):
    try:
        audio = _safe_load_audio(path)
        loudness = audio.dBFS
        if loudness < -30:
            status = "Too Quiet"
        elif loudness > -5:
            status = "Too Loud (Might Clip)"
        else:
            status = "Good Volume"
        return {"status": status, "value_dbfs": round(loudness, 2)}
    except Exception as e:
        return {"error": str(e)}

def calculate_snr(path, noise_duration_ms=1000):
    try:
        audio = _safe_load_audio(path).set_channels(1)
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
        audio = _safe_load_audio(path).set_channels(1)
        silent_ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        total_silence = sum((end - start)/1000.0 for start, end in silent_ranges)
        if total_silence > 5:
            status = "Too much silence"
        elif len(silent_ranges) > 3:
            status = "Frequent silent pauses"
        else:
            status = "Silence level is okay"
        return {"silent_periods": len(silent_ranges), "total_silence_seconds": round(total_silence, 2), "status": status}
    except Exception as e:
        return {"error": str(e)}

def detect_audio_issues(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        clip_ratio = np.sum(np.abs(y) >= 0.99) / max(1, len(y))
        avg_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y + 1e-12)))
        messages = []
        if clip_ratio > 0.01: messages.append("Clipping detected")
        if avg_flatness > 0.4: messages.append("Echo or noise-like signal detected")
        status = "Audio is clean" if not messages else "; ".join(messages)
        return {"clip_ratio": clip_ratio, "avg_spectral_flatness": round(avg_flatness,4), "status": status}
    except Exception as e:
        return {"error": str(e)}

# ---------------- Video Checks ----------------
def detect_black_screen(path, sample_rate=30, black_threshold=0.6):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return {"error": "Could not open video"}
        black_frames, total_sampled, frame_index = 0,0,0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_index % sample_rate == 0:
                if frame is not None and frame.size > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if np.mean(gray) < 10: black_frames += 1
                    total_sampled += 1
            frame_index += 1
        cap.release()
        if total_sampled == 0: return {"error": "Video empty or unreadable"}
        ratio = black_frames / total_sampled
        if ratio > black_threshold: status="Mostly black screen"
        elif ratio > 0.1: status="Some black frames"
        else: status="Video is fine"
        return {"black_frame_ratio": round(ratio,3), "status": status}
    except Exception as e:
        return {"error": str(e)}

def detect_blurriness(path, sample_rate=30, blurry_threshold=50.0):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return {"error": "Could not open video"}
        blurry_frames, total_sampled, frame_index = 0,0,0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_index % sample_rate == 0:
                if frame is not None and frame.size > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if cv2.Laplacian(gray, cv2.CV_64F).var() < blurry_threshold: blurry_frames += 1
                    total_sampled += 1
            frame_index += 1
        cap.release()
        if total_sampled == 0: return {"error": "Video empty or unreadable"}
        ratio = blurry_frames / total_sampled
        if ratio > 0.6: status="Video is too blurry"
        elif ratio > 0.2: status="Some blurriness"
        else: status="Video is sharp enough"
        return {"blurry_frame_ratio": round(ratio,3), "status": status}
    except Exception as e:
        return {"error": str(e)}

def check_resolution(path, min_width=640, min_height=480):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return {"error": "Could not open video"}
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        status = "Resolution is good" if width>=min_width and height>=min_height else f"Resolution too low: {width}x{height}"
        return {"resolution": f"{width}x{height}", "status": status}
    except Exception as e:
        return {"error": str(e)}

def check_frame_rate(path, min_fps=24):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return {"error": "Could not open video"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        status = "Frame rate is good" if fps>=min_fps else "Frame rate is low"
        return {"fps": round(float(fps),2), "status": status}
    except Exception as e:
        return {"error": str(e)}


def verify_identity(reference_image_path, video_path):
    verifier = FaceVerificationSystem(model_name="Facenet", similarity_threshold=0.6)
    return verifier.verify_identity(reference_image_path, video_path)


def detect_face_consistency(video_path, sample_rate=30, min_detection_ratio=0.7):
    try:
        import face_recognition
    except:
        return {"error": "face_recognition not installed"}
    cap = cv2.VideoCapture(video_path)
    total_sampled, face_detected, frame_index = 0,0,0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_index % sample_rate == 0:
            if frame is not None and frame.size > 0:
                total_sampled += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb, model="hog")
                if faces: face_detected += 1
        frame_index += 1
    cap.release()
    if total_sampled == 0: return {"error": "video unreadable"}
    ratio = face_detected / total_sampled
    status = "Face detected consistently" if ratio >= min_detection_ratio else "Face not detected reliably"
    return {"detection_ratio": round(ratio,3), "status": status}

# ---------------- Run All Checks ----------------

def run_all_checks(video_path, reference_image_path=None):
    results = {}
    audio_path = None

    # 1) Extract audio to a temp .wav safely
    try:
        clip = VideoFileClip(video_path)
        if clip.audio:
            tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = tmp_audio.name
            tmp_audio.close()
            clip.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
    except Exception as e:
        results["audio_extract"] = {"error": f"Could not extract audio: {e}"}
    finally:
        # Ensure clip is closed
        try:
            clip.close()
        except Exception:
            pass

    # 2) Run audio checks only if audio was successfully extracted
    if audio_path and os.path.exists(audio_path):
        try:
            results["audio_loudness"] = check_audio_loudness(audio_path)
            results["snr"] = calculate_snr(audio_path)
            results["audio_issues"] = detect_audio_issues(audio_path)
            results["silence"] = detect_silence_periods(audio_path)
        finally:
            # Always clean up temp audio file
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"[WARN] Could not delete temp audio file {audio_path}: {e}")
    else:
        results["audio_checks"] = {"status": "No audio track detected"}

    # 3) Video checks
    results["black_screen"] = detect_black_screen(video_path)
    results["blurriness"] = detect_blurriness(video_path)
    results["resolution"] = check_resolution(video_path)
    results["frame_rate"] = check_frame_rate(video_path)
    results["face_consistency"] = detect_face_consistency(video_path)

    # 4) Identity verification (if reference photo provided)
    if reference_image_path:
        try:
            results["identity_verification"] = verify_identity(reference_image_path, video_path)
        except Exception as e:
            results["identity_verification"] = {"error": str(e)}

    return results

def run_audio_checks(audio_path):
    
    results = {}
    results["loudness"] = check_audio_loudness(audio_path)
    results["snr"] = calculate_snr(audio_path)
    results["silence"] = detect_silence_periods(audio_path)
    results["audio_issues"] = detect_audio_issues(audio_path)
    
    return results

# inside video_checks.py