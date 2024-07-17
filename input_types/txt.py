import re
import nltk
from collections import Counter
from nltk.corpus import stopwords


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


def process_text_file(
    txt_path: str,
    words_per_sentence: int,
    sentences_per_page: int,
    output_file: str = None,
) -> None:
    """Process a text file and optionally save summarized text to a file or print to console."""
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()

    cleaned_text = preprocess_text(text)
    text_no_stop = remove_stop_words(cleaned_text)
    word_freq = get_word_frequencies(text_no_stop)
    sent_scores = score_sentences(text, word_freq, words_per_sentence)

    output_lines = []

    summary = summarize_text(sent_scores, sentences_per_page)

    output_lines.append(f"\nTLDR:\n")
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
