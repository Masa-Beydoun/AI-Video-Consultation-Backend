from django.conf import settings
import os, uuid
from moviepy import VideoFileClip
from transformers import pipeline
import torch
from .segmenter import VideoQASegmenter

import logging
logger = logging.getLogger(__name__)

def segment_video_into_consultations(video_path, model_dir="./qa_classifier"):
    segmenter = VideoQASegmenter(language="multilingual")
    segmenter.classifier_model = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir,
        device=0 if torch.cuda.is_available() else -1,
    )

    results = segmenter.process_video(video_path)

    # 🔹 Pair Q/A
    qa_pairs = pair_segments_into_consultations(results)
    
    # ✅ Save under MEDIA_ROOT/segments
    out_dir = os.path.join(settings.MEDIA_ROOT, "segments")
    os.makedirs(out_dir, exist_ok=True)

    output_segments = []
    clip = VideoFileClip(video_path)

    for qa in qa_pairs:
        out_name = f"{uuid.uuid4().hex}.mp4"
        out_path = os.path.join(out_dir, out_name)

        subclip = clip.subclipped(qa["start"], qa["end"])  
        subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")

        # Optional: store relative path for serving via MEDIA_URL
        qa["file_path"] = os.path.relpath(out_path, settings.MEDIA_ROOT)
        output_segments.append(qa)

    clip.close()
    logger.info("Generated segments:")
    for seg in output_segments:
        logger.info(seg)
    return output_segments

def pair_segments_into_consultations(segments):
    """
    Merge consecutive questions into one and consecutive answers into one.
    Produces a clean sequence of Q/A pairs.
    """
    consultations = []
    current_question = None
    current_answer = None
    start_time, end_time = None, None

    for seg in segments:
        if seg["label"] == "question":
            # If we already had a question and some answers, finalize that consultation
            if current_question and current_answer:
                consultations.append({
                    "question": current_question.strip(),
                    "answer": current_answer.strip(),
                    "start": start_time,
                    "end": end_time,
                })
                current_question, current_answer = None, None
                start_time, end_time = None, None

            # If still accumulating questions
            if not current_question:
                current_question = seg["text"]
                start_time = seg["start"]
            else:
                current_question += " " + seg["text"]

            end_time = seg["end"]

        elif seg["label"] == "answer":
            if not current_question:
                # Edge case: answer without question
                current_question = ""
                start_time = seg["start"]

            # If still accumulating answers
            if not current_answer:
                current_answer = seg["text"]
            else:
                current_answer += " " + seg["text"]

            end_time = seg["end"]

    # Final flush
    if current_question and current_answer:
        consultations.append({
            "question": current_question.strip(),
            "answer": current_answer.strip(),
            "start": start_time,
            "end": end_time,
        })

    return consultations
