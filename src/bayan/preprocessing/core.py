"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""
import re

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    # Remove Arabic Tatweel
    text = text.replace("\u0640", "")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Reduce any character repeated 3+ times to 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""

    # Saudi mobile numbers:
    # 05XXXXXXXX
    # +9665XXXXXXXX
    # 9665XXXXXXXX
    text = re.sub(
        r"(?<!\d)(?:\+966|966|0)?5\d{8}(?!\d)",
        "<PHONE>",
        text,
    )

    # Saudi national-ID-shaped values
    text = re.sub(
        r"(?<!\d)\d{10}(?!\d)",
        "<NATIONAL_ID>",
        text,
    )

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    return normalize(mask_pii(text))