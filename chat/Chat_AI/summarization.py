from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # You can also try LexRankSummarizer, LuhnSummarizer, etc.
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

def summarize_text(text, sentences_count=3):
    """
    Summarizes the given text into a specified number of sentences.
    
    :param text: Input text to summarize
    :param sentences_count: Number of sentences in the summary
    :return: Summary string
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)


if __name__ == "__main__":
    # 🔹 Replace this with your text
    text = """
    Artificial Intelligence (AI) is transforming industries across the globe. 
    It is used in healthcare, finance, education, and transportation. 
    AI systems can process large amounts of data faster than humans, 
    making predictions and automating tasks. 
    However, ethical concerns such as privacy, bias, and job displacement remain critical challenges. 
    Governments and organizations are now working to regulate AI to ensure safe and fair use.
    """

    summary = summarize_text(text, sentences_count=3)
    print("\n--- Summary ---")
    print(summary)
