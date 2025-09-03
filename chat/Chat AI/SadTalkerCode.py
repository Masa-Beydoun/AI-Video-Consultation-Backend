import os
import subprocess

def generate_video_with_driving_video(face_image_path, driving_video_path, output_video_path):
    SADTALKER_ROOT = r"D:\VideoGenerating\SadTalker"
    inference_script = os.path.join(SADTALKER_ROOT, "inference.py")  # typical SadTalker script

    cmd = [
        "python", inference_script,
        "--source_img", face_image_path,
        "--driving_video", driving_video_path,
        "--output_dir", os.path.dirname(output_video_path),
        "--device", "cuda"
    ]

    print("Running SadTalker...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("SadTalker failed.")
    else:
        print("Done.")


if __name__ == "__main__":
    face_img = r"C:\Users\DELL\OneDrive\Desktop\New folder\photo_2025-06-28_06-59-49.jpg"      # Path to your face image
    audio_file = "audio.mp3"   # Path to your generated TTS audio file
    output_vid = "result.mp4"  # Desired output video file

    generate_video_with_driving_video(face_img, audio_file, output_vid)
