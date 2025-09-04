# faq_matcher.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import re
from typing import List, Dict, Tuple, Optional, Union
import spacy
from collections import defaultdict

# ==============================
# NLP Setup
# ==============================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# ==============================
# Question Splitter
# ==============================
class QuestionSplitter:
    def __init__(self):
        self.conjunction_patterns = [
            r'\band\b', r'\bor\b', r'\bthen\b', r'\balso\b',
            r'\bplus\b', r';', r','
        ]

    def split_questions(self, text: str) -> List[str]:
        """Split text into multiple questions by '?' and conjunctions."""
        questions = []
        parts = re.split(r'\?', text)

        for part in parts[:-1]:  # skip last empty
            question = part.strip()
            if question:
                question += "?"
                questions.extend(self._split_by_conjunctions(question))

        return [q for q in questions if self._is_valid_question(q)] or [text.strip()]

    def _split_by_conjunctions(self, question: str) -> List[str]:
        if len(question.split()) > 15:
            for pattern in self.conjunction_patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    parts = re.split(pattern, question, 1)
                    if len(parts) == 2:
                        return [parts[0].strip() + "?", parts[1].strip() + "?"]
        return [question]

    def _is_valid_question(self, text: str) -> bool:
        if len(text) < 5:
            return False
        q_words = ['what', 'how', 'when', 'where', 'why', 'who',
                   'which', 'can', 'could', 'would', 'should',
                   'is', 'are', 'do', 'does', 'did']
        return text.endswith("?") or any(w in text.lower().split()[:3] for w in q_words)

# ==============================
# Conversation Context
# ==============================
class ConversationContext:
    def __init__(self):
        self.history = []

    def add_interaction(self, question: str, answer: str, entities: Dict = None):
        self.history.append({
            "question": question,
            "answer": answer,
            "entities": entities or {}
        })

    def get_recent_entities(self) -> List[str]:
        ents = []
        for h in reversed(self.history[-3:]):  # last 3
            for v in h["entities"].values():
                ents.extend(v)
        return ents

    def clear(self):
        self.history = []

# ==============================
# Follow-up Processor
# ==============================
class FollowUpProcessor:
    def __init__(self):
        self.pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']

    def extract_entities(self, text: str) -> Dict:
        entities = {}
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    def resolve(self, question: str, context: ConversationContext) -> str:
        if not context.history:
            return question
        resolved = question
        ents = context.get_recent_entities()
        if ents:
            for pronoun in self.pronouns:
                resolved = re.sub(rf"\b{pronoun}\b", ents[0], resolved, flags=re.IGNORECASE)
        return resolved

# ==============================
# Domain Classifier
# ==============================
class DomainClassifier:
    def __init__(self, model: str = "facebook/bart-large-mnli"):
        self.pipe = pipeline("zero-shot-classification", model=model,
                             device=0 if torch.cuda.is_available() else -1)

    def is_relevant(self, text: str, domain: str, threshold: float = 0.7) -> bool:
        labels = [f"This text is about {domain}"]
        res = self.pipe(text, labels)
        return res["scores"][0] > threshold

# ==============================
# FAQ Matcher
# ==============================
def match_question(user_q, model, faqs, faq_embeddings,
                   threshold=0.65, suggestion_threshold=0.4, max_suggestions=3):
    u_emb = model.encode(user_q, convert_to_tensor=True)
    scores = cos_sim(u_emb, faq_embeddings)[0]
    sorted_idx = torch.argsort(scores, descending=True)

    result = {"matched": False, "main_id": None, "suggestions": []}

    if scores[sorted_idx[0]].item() >= threshold:
        i = sorted_idx[0].item()
        result["matched"] = True
        result["main"] = {
            "id": faqs[i]["consultation_id"],
            "answer": faqs[i]["answer"],
            "score": float(scores[sorted_idx[0]])
        }

    for j in sorted_idx[1:max_suggestions+1]:
        sc = scores[j].item()
        if sc >= suggestion_threshold:
            result["suggestions"].append({
                "id": faqs[j.item()]["consultation_id"],
                "answer": faqs[j.item()]["answer"],
                "score": float(sc)
            })
    return result

# ==============================
# MultiQuestionHandler
# ==============================
class MultiQuestionHandler:
    def __init__(self, faqs: List[Dict]):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.classifier = DomainClassifier()
        self.faqs = faqs
        self.faq_embeddings = self.model.encode(
            [f["question"] for f in faqs], convert_to_tensor=True
        )
        self.splitter = QuestionSplitter()
        self.context = ConversationContext()
        self.followup = FollowUpProcessor()

    def process(self, user_input: str, domain: str = "general") -> Dict:
        qs = self.splitter.split_questions(user_input)
        out = {"input": user_input, "results": []}
        for q in qs:
            resolved = self.followup.resolve(q, self.context)
            ents = self.followup.extract_entities(resolved)
            match = match_question(resolved, self.model, self.faqs, self.faq_embeddings)
            out["results"].append({
                "original": q,
                "resolved": resolved,
                "entities": ents,
                "match": match  # contains only IDs now
            })
            if match["matched"]:
                # still update context for follow-ups
                self.context.add_interaction(resolved, str(match["main_id"]), ents)
        return out

    def clear(self):
        self.context.clear()

