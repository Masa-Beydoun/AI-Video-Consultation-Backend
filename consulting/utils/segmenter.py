# !python -m spacy download en_core_web_sm
# !python -m spacy download ar_core_news_sm
# Video Question-Answer Segmentation Pipeline
# Supports Arabic and English text processing

# Installation requirements (run in terminal):
# pip install openai-whisper torch transformers datasets scikit-learn arabic-reshaper python-bidi spacy
# python -m spacy download en_core_web_sm
# python -m spacy download ar_core_news_sm

import whisper
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from datasets import Dataset
import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import spacy
from typing import List, Dict, Tuple
import arabic_reshaper
from bidi.algorithm import get_display
import warnings
warnings.filterwarnings('ignore')

class VideoQASegmenter:
    def __init__(self, language='multilingual'):
        """
        Initialize the Video Question-Answer Segmenter

        Args:
            language: 'arabic', 'english', or 'multilingual'
        """
        self.language = language
        self.whisper_model = None
        self.classifier_model = None
        self.tokenizer = None
        self.nlp = None

        # Load spaCy models
        try:
            if language == 'english' or language == 'multilingual':
                self.nlp_en = spacy.load('en_core_web_sm')
            # Arabic model fallback - use English model for sentence segmentation
            if language == 'arabic' or language == 'multilingual':
                try:
                    self.nlp_ar = spacy.load('ar_core_news_sm')
                except OSError:
                    print("Arabic spaCy model not available, using English model for sentence segmentation")
                    self.nlp_ar = self.nlp_en if hasattr(self, 'nlp_en') else None
        except OSError:
            print("Please install English spaCy model: python -m spacy download en_core_web_sm")

    def load_whisper_model(self, model_size='base'):
        """Load Whisper model for speech-to-text"""
        print(f"Loading Whisper {model_size} model...")
        self.whisper_model = whisper.load_model(model_size)
        print("Whisper model loaded successfully!")

    def transcribe_video(self, video_path: str) -> Dict:
        """
        Transcribe video to text with word-level timestamps

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with transcription and word timestamps
        """
        if self.whisper_model is None:
            self.load_whisper_model()

        print(f"Transcribing video: {video_path}")
        result = self.whisper_model.transcribe(
            video_path,
            word_timestamps=True,
            language=None  # Auto-detect language
        )

        return result

    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        return 'arabic' if arabic_chars > english_chars else 'english'

    def preprocess_text(self, text: str, lang: str = None) -> str:
        """Preprocess text based on language"""
        if lang is None:
            lang = self.detect_language(text)

        # Common preprocessing
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        if lang == 'arabic':
            # Arabic text preprocessing
            text = arabic_reshaper.reshape(text)
            text = get_display(text)

        return text

    def create_training_data(self) -> pd.DataFrame:
        """
        Create synthetic training data for question-answer classification
        This is a basic example - you should expand this with real data
        """

        # Sample training data (English)
        english_data = [
            # Questions
            ("What is machine learning?", "question"),
            ("How does neural network work?", "question"),
            ("Why is data preprocessing important?", "question"),
            ("When should we use deep learning?", "question"),
            ("Where can we apply AI?", "question"),
            ("Which algorithm is best for classification?", "question"),
            ("Who invented the backpropagation algorithm?", "question"),
            ("Can you explain supervised learning?", "question"),
            ("Is this a good approach?", "question"),
            ("Do you understand the concept?", "question"),

            # Answers
            ("Machine learning is a subset of artificial intelligence.", "answer"),
            ("It uses algorithms to learn patterns from data automatically.", "answer"),
            ("Neural networks consist of interconnected nodes or neurons.", "answer"),
            ("Data preprocessing helps improve model accuracy significantly.", "answer"),
            ("Deep learning works well with large datasets and complex problems.", "answer"),
            ("AI applications include healthcare, finance, and autonomous vehicles.", "answer"),
            ("The choice depends on your specific problem and data characteristics.", "answer"),
            ("Backpropagation was developed by multiple researchers in the 1970s-80s.", "answer"),
            ("Supervised learning uses labeled data to train predictive models.", "answer"),
            ("Yes, this approach has proven effective in many real-world scenarios.", "answer"),
        ]

        # Sample training data (Arabic)
        arabic_data = [
            # Questions (Arabic)
            ("ما هو التعلم الآلي؟", "question"),
            ("كيف تعمل الشبكات العصبية؟", "question"),
            ("لماذا معالجة البيانات مهمة؟", "question"),
            ("متى نستخدم التعلم العميق؟", "question"),
            ("أين يمكن تطبيق الذكاء الاصطناعي؟", "question"),
            ("أي خوارزمية الأفضل للتصنيف؟", "question"),
            ("من اخترع خوارزمية الانتشار العكسي؟", "question"),
            ("هل يمكنك شرح التعلم المراقب؟", "question"),
            ("هل هذا منهج جيد؟", "question"),
            ("هل تفهم المفهوم؟", "question"),

            # Answers (Arabic)
            ("التعلم الآلي هو فرع من الذكاء الاصطناعي.", "answer"),
            ("يستخدم خوارزميات لتعلم الأنماط من البيانات تلقائياً.", "answer"),
            ("الشبكات العصبية تتكون من عقد مترابطة.", "answer"),
            ("معالجة البيانات تساعد في تحسين دقة النموذج.", "answer"),
            ("التعلم العميق يعمل جيداً مع البيانات الكبيرة والمشاكل المعقدة.", "answer"),
            ("تطبيقات الذكاء الاصطناعي تشمل الصحة والمالية والمركبات الذاتية.", "answer"),
            ("الاختيار يعتمد على مشكلتك المحددة وخصائص البيانات.", "answer"),
            ("الانتشار العكسي طوره عدة باحثين في السبعينات والثمانينات.", "answer"),
            ("التعلم المراقب يستخدم بيانات موسومة لتدريب النماذج التنبؤية.", "answer"),
            ("نعم، هذا المنهج أثبت فعاليته في العديد من السيناريوهات الحقيقية.", "answer"),
        ]

        # Combine data
        all_data = english_data + arabic_data

        df = pd.DataFrame(all_data, columns=['text', 'label'])
        return df

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts"""
        features = []

        for text in texts:
            lang = self.detect_language(text)

            # Basic features
            feature_vector = [
                len(text),  # text length
                len(text.split()),  # word count
                text.count('?'),  # question marks
                text.count('.'),  # periods
                text.count('!'),  # exclamation marks
                len(re.findall(r'[A-Z]', text)),  # capital letters
                len(re.findall(r'[\u0600-\u06FF]', text)),  # Arabic characters
            ]

            # Language-specific question words
            if lang == 'english':
                question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'is', 'do', 'does', 'did', 'will', 'would', 'could', 'should']
                for word in question_words:
                    feature_vector.append(1 if word.lower() in text.lower() else 0)
            elif lang == 'arabic':
                question_words = ['ما', 'كيف', 'لماذا', 'متى', 'أين', 'أي', 'من', 'هل', 'ماذا', 'كم']
                for word in question_words:
                    feature_vector.append(1 if word in text else 0)
            else:
                # Pad with zeros if language not detected
                feature_vector.extend([0] * 16)

            # Pad or truncate to fixed size
            while len(feature_vector) < 23:
                feature_vector.append(0)

            features.append(feature_vector[:23])

        return np.array(features)

    def train_classifier(self, train_data: pd.DataFrame):
        """Train a transformer-based classifier for question/answer detection"""

        # Use multilingual BERT for both Arabic and English
        model_name = "bert-base-multilingual-cased"

        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512
            )

        # Convert labels to numbers
        train_data['labels'] = train_data['label'].map({'question': 0, 'answer': 1})

        # Create train/validation split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data['text'].tolist(),
            train_data['labels'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=train_data['labels'].tolist()
        )

        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })

        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })

        # Tokenize datasets
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            eval_strategy="epoch",  # Updated parameter name
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        print("Training classifier...")
        trainer.train()

        # Save the trained model
        self.classifier_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        print("Classifier trained successfully!")

    def segment_sentences(self, text: str, word_timestamps: List[Dict]) -> List[Dict]:
        """
        Segment text into sentences with timestamps

        Args:
            text: Full transcribed text
            word_timestamps: List of word-level timestamps from Whisper

        Returns:
            List of sentence segments with timestamps
        """

        # Detect language
        lang = self.detect_language(text)

        # Use appropriate spaCy model
        if lang == 'arabic' and hasattr(self, 'nlp_ar') and self.nlp_ar is not None:
            nlp = self.nlp_ar
        elif hasattr(self, 'nlp_en') and self.nlp_en is not None:
            nlp = self.nlp_en
        else:
            # Fallback: simple sentence splitting
            sentences = self._simple_sentence_split(text)
            return self._map_simple_sentences_to_timestamps(sentences, word_timestamps)

        # Process text with spaCy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        # Map sentences to timestamps
        sentence_segments = []
        word_idx = 0

        for sentence in sentences:
            if not sentence:
                continue

            sentence_words = sentence.split()
            start_time = None
            end_time = None

            # Find timestamps for this sentence
            for _ in sentence_words:
                if word_idx < len(word_timestamps):
                    word_info = word_timestamps[word_idx]
                    if start_time is None:
                        start_time = word_info.get('start', 0.0)
                    end_time = word_info.get('end', 0.0)
                    word_idx += 1

            if start_time is not None and end_time is not None:
                sentence_segments.append({
                    'text': sentence,
                    'start': start_time,
                    'end': end_time
                })

        return sentence_segments

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback when spaCy is not available"""
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _map_simple_sentences_to_timestamps(self, sentences: List[str], word_timestamps: List[Dict]) -> List[Dict]:
        """Map simple sentences to timestamps when spaCy is not available"""
        sentence_segments = []
        word_idx = 0

        for sentence in sentences:
            if not sentence:
                continue

            sentence_words = sentence.split()
            start_time = None
            end_time = None

            # Find timestamps for this sentence
            for _ in sentence_words:
                if word_idx < len(word_timestamps):
                    word_info = word_timestamps[word_idx]
                    if start_time is None:
                        start_time = word_info.get('start', 0.0)
                    end_time = word_info.get('end', 0.0)
                    word_idx += 1

            if start_time is not None and end_time is not None:
                sentence_segments.append({
                    'text': sentence,
                    'start': start_time,
                    'end': end_time
                })

        return sentence_segments

    def classify_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Classify segments as questions or answers

        Args:
            segments: List of text segments with timestamps

        Returns:
            List of segments with labels
        """
        if self.classifier_model is None:
            raise ValueError("Classifier not trained. Please train the classifier first.")

        labeled_segments = []

        for segment in segments:
            text = segment['text']

            # Get prediction
            prediction = self.classifier_model(text)[0]
            label = 'question' if prediction['label'] == 'LABEL_0' else 'answer'
            confidence = prediction['score']

            labeled_segments.append({
                'text': text,
                'label': label,
                'start': segment['start'],
                'end': segment['end'],
                'confidence': confidence
            })

        return labeled_segments

    def post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Post-process segments to ensure logical question-answer flow
        """
        processed_segments = []

        for i, segment in enumerate(segments):
            # Rule-based corrections
            text = segment['text'].strip()

            # If text ends with '?', it's likely a question
            if text.endswith('?'):
                segment['label'] = 'question'

            # If text starts with question words, it's likely a question
            question_patterns_en = r'^(what|how|why|when|where|which|who|can|is|do|does|did|will|would|could|should)\b'
            question_patterns_ar = r'^(ما|كيف|لماذا|متى|أين|أي|من|هل|ماذا|كم)\b'

            if re.search(question_patterns_en, text.lower()) or re.search(question_patterns_ar, text):
                segment['label'] = 'question'

            processed_segments.append(segment)

        return processed_segments

    def process_video(self, video_path: str) -> List[Dict]:
        """
        Complete pipeline to process video and extract question-answer pairs

        Args:
            video_path: Path to video file

        Returns:
            List of question-answer segments with timestamps
        """

        # Step 1: Transcribe video
        print("Step 1: Transcribing video...")
        transcription_result = self.transcribe_video(video_path)

        full_text = transcription_result['text']
        word_timestamps = []

        # Extract word timestamps
        for segment in transcription_result['segments']:
            if 'words' in segment:
                word_timestamps.extend(segment['words'])

        print(f"Transcription completed. Text length: {len(full_text)} characters")

        # Step 2: Segment into sentences
        print("Step 2: Segmenting into sentences...")
        sentence_segments = self.segment_sentences(full_text, word_timestamps)
        print(f"Found {len(sentence_segments)} sentence segments")

        # Step 3: Classify segments
        print("Step 3: Classifying segments...")
        labeled_segments = self.classify_segments(sentence_segments)

        # Step 4: Post-process
        print("Step 4: Post-processing...")
        final_segments = self.post_process_segments(labeled_segments)

        return final_segments

    def save_results(self, segments: List[Dict], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")

    def display_results(self, segments: List[Dict]):
        """Display results in a readable format"""
        print("\n" + "="*80)
        print("QUESTION-ANSWER SEGMENTATION RESULTS")
        print("="*80)

        for i, segment in enumerate(segments):
            label_emoji = "❓" if segment['label'] == 'question' else "💡"
            print(f"\n{i+1}. {label_emoji} [{segment['label'].upper()}] ({segment['start']:.1f}s - {segment['end']:.1f}s)")
            print(f"   {segment['text']}")
            if 'confidence' in segment:
                print(f"   Confidence: {segment['confidence']:.3f}")

# Example usage and demonstration
def main():
    # Initialize the segmenter
    segmenter = VideoQASegmenter(language='multilingual')

    # Create and train classifier with sample data
    print("Creating training data...")
    train_data = segmenter.create_training_data()
    print(f"Training data created with {len(train_data)} samples")
    print("\nSample training data:")
    print(train_data.head())

    print("\nTraining classifier...")
    segmenter.train_classifier(train_data)

    # Example: Process a video file
    # Replace 'path/to/your/video.mp4' with actual video path
    video_path = "IMG_7003.mp4"

    # Uncomment below lines when you have a video file:
    print(f"\nProcessing video: {video_path}")
    results = segmenter.process_video(video_path)
    segmenter.display_results(results)
    segmenter.save_results(results, "qa_segments.json")

    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("To process your videos:")
    print("1. Replace 'path/to/your/video.mp4' with your actual video path")
    print("2. Uncomment the processing lines in main()")
    print("3. Run the cell again")
    print("\nNote: Arabic spaCy model not available - using fallback sentence segmentation")
    print("For better Arabic support, you can implement custom Arabic sentence segmentation")
    print("\nSupported formats: MP4, AVI, MOV, WAV, MP3")
    print("Supported languages: Arabic and English")

if __name__ == "__main__":
    main()

# Additional utility functions for batch processing
def process_multiple_videos(video_paths: List[str], output_dir: str = "outputs"):
    """Process multiple videos in batch"""
    import os

    os.makedirs(output_dir, exist_ok=True)
    segmenter = VideoQASegmenter(language='multilingual')

    # Train classifier once
    train_data = segmenter.create_training_data()
    segmenter.train_classifier(train_data)

    results = {}

    for video_path in video_paths:
        print(f"\nProcessing: {video_path}")

        try:
            segments = segmenter.process_video(video_path)

            # Save individual results
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}_qa_segments.json")
            segmenter.save_results(segments, output_path)

            results[video_path] = segments

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results[video_path] = None

    return results

# Function to improve model with your own data
def fine_tune_with_custom_data(custom_data_path: str):
    """
    Fine-tune the model with your custom question-answer data

    Expected CSV format:
    text,label
    "What is AI?",question
    "AI is artificial intelligence",answer
    """

    # Load custom data
    custom_df = pd.read_csv(custom_data_path)

    # Initialize segmenter
    segmenter = VideoQASegmenter(language='multilingual')

    # Combine with base training data
    base_data = segmenter.create_training_data()
    combined_data = pd.concat([base_data, custom_df], ignore_index=True)

    print(f"Training with {len(combined_data)} total samples ({len(custom_df)} custom)")

    # Train classifier
    segmenter.train_classifier(combined_data)

    return segmenter