import re
import nltk
import PyPDF2
from collections import Counter
from nltk.corpus import stopwords


def extract_text_from_pdf(pdf_path: str) -> list:
    """Extract text from each page of a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return [page.extract_text() for page in reader.pages if page.extract_text()]


def preprocess_text(text: str) -> str:
    """Clean and preprocess the text."""
    text = re.sub(r"\s+", " ", text)
    return re.sub("[^a-zA-Z]", " ", text)


def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    return " ".join(word for word in words if word.lower() not in stop_words)


def get_word_frequencies(text: str) -> dict:
    """Get the frequency of each word in the text."""
    words = nltk.word_tokenize(text.lower())
    return nltk.FreqDist(words)


def score_sentences(text: str, word_freq: dict, max_words: int) -> dict:
    """Score sentences based on word frequencies."""
    sentences = nltk.sent_tokenize(text)
    sent_scores = {}
    for sent in sentences:
        words = nltk.word_tokenize(sent.lower())
        if len(words) <= max_words:
            score = sum(word_freq.get(word, 0) for word in words) / len(words)
            sent_scores[sent] = score
    return sent_scores


def summarize_text(sent_scores: dict, num_sents: int) -> list:
    """Summarize the text by selecting the top scoring sentences."""
    counts = Counter(sent_scores)
    return counts.most_common(num_sents)


def process_pdf(
    pdf_path: str,
    words_per_sentence: int,
    sentences_per_page: int,
    output_file: str = None,
) -> None:
    """Process a PDF file and optionally save summarized text to a file or print to console."""
    pages_text = extract_text_from_pdf(pdf_path)
    output_lines = []

    for page_num, page_text in enumerate(pages_text, start=1):
        if not page_text:
            output_lines.append("")
            continue

        cleaned_text = preprocess_text(page_text)
        text_no_stop = remove_stop_words(cleaned_text)
        word_freq = get_word_frequencies(text_no_stop)
        sent_scores = score_sentences(page_text, word_freq, words_per_sentence)

        if not sent_scores:
            output_lines.append("")
            continue

        summary = summarize_text(sent_scores, sentences_per_page)

        output_lines.append(f"\nPage {page_num}\n")
        if summary:
            for sent, score in summary:
                output_lines.append(sent + "\n")
        else:
            output_lines.append("No sentences found.\n")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.writelines(output_lines)
        print(f"TLDR saved to {output_file}")
    else:
        for line in output_lines:
            print(line, end="")
