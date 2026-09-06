"""Lab 1 starter: sentence segmentation."""

import spacy

from .core import preprocess


def build_pipeline():
    """Build a lightweight spaCy sentence-segmentation pipeline."""
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess text, then return non-empty sentence strings."""
    text = preprocess(raw)

    if not text:
        return []

    doc = nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]