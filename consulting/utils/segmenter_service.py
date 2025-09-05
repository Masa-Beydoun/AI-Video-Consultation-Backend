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

    # 🔹 Pair Q/A (now contains separate answer timestamps)
    qa_pairs = pair_segments_into_consultations(results)
    
    # ✅ Save under MEDIA_ROOT/segments
    out_dir = os.path.join(settings.MEDIA_ROOT, "segments")
    os.makedirs(out_dir, exist_ok=True)

    output_segments = []
    clip = VideoFileClip(video_path)

    for qa in qa_pairs:
        out_name = f"{uuid.uuid4().hex}.mp4"
        out_path = os.path.join(out_dir, out_name)

        # ⏱️ Only cut the answer part
        subclip = clip.subclipped(qa["answer_start"], qa["answer_end"])
        subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")

        qa["file_path"] = os.path.relpath(out_path, settings.MEDIA_ROOT)
        output_segments.append(qa)

    clip.close()
    logger.info("Generated segments:")
    for seg in output_segments:
        logger.info(seg)
    return output_segments


def pair_segments_into_consultations(segments):
    consultations = []
    current_question, current_answer = None, None
    question_confidences, answer_confidences = [], []
    question_start, question_end = None, None
    answer_start, answer_end = None, None

    for seg in segments:
        if seg["label"] == "question":
            # finalize previous pair
            if current_question and current_answer:
                consultations.append({
                    "question": current_question.strip(),
                    "answer": current_answer.strip(),
                    "confidence_question": sum(question_confidences)/len(question_confidences) if question_confidences else None,
                    "confidence_answer": sum(answer_confidences)/len(answer_confidences) if answer_confidences else None,
                    "question_start": question_start,
                    "question_end": question_end,
                    "answer_start": answer_start,
                    "answer_end": answer_end,
                })
                current_question, current_answer = None, None
                question_confidences, answer_confidences = [], []
                question_start, question_end, answer_start, answer_end = None, None, None, None

            # start new question
            current_question = (current_question + " " if current_question else "") + seg["text"]
            if "confidence" in seg:
                question_confidences.append(seg["confidence"])
            if question_start is None:
                question_start = seg["start"]
            question_end = seg["end"]

        elif seg["label"] == "answer":
            current_answer = (current_answer + " " if current_answer else "") + seg["text"]
            if "confidence" in seg:
                answer_confidences.append(seg["confidence"])
            if answer_start is None:
                answer_start = seg["start"]
            answer_end = seg["end"]

    # flush last one
    if current_question and current_answer:
        consultations.append({
            "question": current_question.strip(),
            "answer": current_answer.strip(),
            "confidence_question": sum(question_confidences)/len(question_confidences) if question_confidences else None,
            "confidence_answer": sum(answer_confidences)/len(answer_confidences) if answer_confidences else None,
            "question_start": question_start,
            "question_end": question_end,
            "answer_start": answer_start,
            "answer_end": answer_end,
        })

    return consultations
