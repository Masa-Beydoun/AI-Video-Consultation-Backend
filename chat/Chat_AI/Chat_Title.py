from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_chat_title(chat_messages):
    chat_text = " ".join(chat_messages)
    summary = summarizer(chat_text, max_length=20, min_length=5, do_sample=False)[0]['summary_text']
    title = summary.strip()
    return title

if __name__ == "__main__":
    chat_example = [
        "Hey Alex, did you get the project brief from the client?",
        "Yes, I just received it this morning.",
        "Great, what are the key requirements?",
        "They want a mobile app for booking doctor appointments.",
        "Sounds straightforward, any special features?",
        "Yes, they need multi-language support, offline mode, and payment integration.",
        "Okay, what’s the deadline?",
        "They expect a prototype in 3 weeks and full delivery in 2 months.",
        "Do we have a budget confirmed?",
        "Yes, around $15,000 for phase one.",
        "Alright, should we use Flutter or React Native?",
        "Flutter — the client prefers a single codebase and better performance.",
        "Got it. I’ll prepare a feature list and initial design mockups.",
        "Perfect, also make sure we add user authentication with email and Google login.",
        "Noted. Anything else?",
        "The client also mentioned they want a dark mode option.",
        "Cool. Let's set up a kickoff meeting for Monday at 10 AM.",
        "I’ll send the invite. Anything about backend?",
        "Yes, we’ll use Firebase for the MVP to save time.",
        "Alright, I’ll get started on the architecture."
    ]

    print("Generated title:", generate_chat_title(chat_example))
