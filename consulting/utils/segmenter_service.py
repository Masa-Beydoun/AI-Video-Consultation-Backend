# consulting/utils/segmenter_service.py
from consulting.utils.segmenter import VideoQASegmenter

def segment_video_into_consultations(video_path: str):
    """
    Run the Q/A segmentation pipeline and return structured results.
    Consecutive segments of the same type (question/answer) are merged.
    """
    segmenter = VideoQASegmenter(language='multilingual')

    # Ensure classifier is trained
    train_data = segmenter.create_training_data()
    segmenter.train_classifier(train_data)

    results = segmenter.process_video(video_path)

    # Merge consecutive segments of the same label
    merged_segments = []
    for seg in results:
        if not merged_segments:
            merged_segments.append(seg.copy())
            continue

        last = merged_segments[-1]
        if seg["label"] == last["label"]:
            # Merge text
            last["text"] += " " + seg["text"]
            # Update start/end times
            last["start"] = min(last["start"], seg["start"])
            last["end"] = max(last["end"], seg["end"])
            # Average confidence
            last["confidence"] = (last.get("confidence", 0) + seg.get("confidence", 0)) / 2
        else:
            merged_segments.append(seg.copy())

    # Group merged segments into Q/A pairs
    qa_pairs = []
    current_question = None

    for seg in merged_segments:
        if seg["label"] == "question":
            current_question = seg
        elif seg["label"] == "answer" and current_question:
            qa_pairs.append({
                "question": current_question["text"],
                "question_end": current_question["end"],
                "answer": seg["text"],
                "answer_start": seg["start"],
                "confidence_question": current_question.get("confidence", 0),
                "confidence_answer": seg.get("confidence", 0),
            })
            current_question = None

    return qa_pairs
