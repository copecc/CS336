import fasttext
import nltk
import regex as re

from typing import Any
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    Extract plain text from HTML bytes.
    """
    encoding = detect_encoding(html_bytes)
    html_text = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html_text)


def identify_language(text: str, model_path: str = None) -> tuple[Any, float]:
    """
    Identify the language of the given text. Return without the padding `__label__`.
    """
    model_path = model_path or "cs336_data/assets/lid.176.bin"
    model = fasttext.load_model(model_path)
    labels, scores = model.predict(text.split("\n", 1)[0])  # predict on the first line only
    return labels[0].removeprefix("__label__"), scores[0]


def mask_emails(text: str) -> tuple[str, int]:
    """
    Mask email addresses in the text with a placeholder.
    """
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)
    email_count = len(re.findall(email_pattern, text))
    return masked_text, email_count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Mask phone numbers in the text with a placeholder.
    """
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"
    masked_text = re.sub(phone_pattern, "|||PHONE_NUMBER|||", text)
    phone_count = len(re.findall(phone_pattern, text))
    return masked_text, phone_count


def mask_ips(text: str) -> tuple[str, int]:
    """
    Mask IP addresses in the text with a placeholder.
    """
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    masked_text = re.sub(ip_pattern, "|||IP_ADDRESS|||", text)
    ip_count = len(re.findall(ip_pattern, text))
    return masked_text, ip_count


def classify_nsfw(text: str, model_path: str = None) -> tuple[Any, float]:
    model_path = model_path or "cs336_data/assets/dolma_fasttext_nsfw_jigsaw_model.bin"
    model = fasttext.load_model(model_path)
    labels, scores = model.predict(text)
    return labels[0].removeprefix("__label__"), scores[0]


def classify_toxic_speech(text: str, model_path: str = None) -> tuple[Any, float]:
    model_path = model_path or "cs336_data/assets/dolma_fasttext_hatespeech_jigsaw_model.bin"
    model = fasttext.load_model(model_path)
    labels, scores = model.predict(text)
    return labels[0].removeprefix("__label__"), scores[0]


def gopher_quality_filter(text: str) -> bool:
    """
    Apply the Gopher quality filter to the text.
    """
    words = nltk.word_tokenize(text)
    n_words = len(words)
    avg_word_len = (sum(len(word) for word in words) / n_words) if words else 0
    lines = text.splitlines()
    n_lines = len(lines)
    ellipsis_lines = sum(1 for line in lines if line.endswith("..."))
    alpha_words = sum(1 for word in words if re.search(r"[a-zA-Z]", word))

    if n_words < 50 or n_words > 100000:
        # Contain less than 50 or more than 100,000 words.
        return False
    if avg_word_len < 3 or avg_word_len > 10:
        # Have a mean word length outside the range of 3 to 10 characters.
        return False
    if ellipsis_lines / n_lines > 0.3:
        # Have more than 30% of lines ending with an ellipsis ("..."").
        return False
    if alpha_words / n_words < 0.8:
        # Have less than 80% of words containing at least one alphabetic character.
        return False

    return True


def classify_quality(text: str, model_path: str = None) -> tuple[Any, float]:
    """
    Classify the quality of the text.
    """
    model_path = model_path
    model = fasttext.load_model(model_path)
    labels, scores = model.predict(text)
    return labels[0].removeprefix("__label__"), scores[0]
