"""Lab 1: audit four tokenizer candidates on Bayan AR/EN text."""

from pathlib import Path
import csv
import math

from transformers import AutoTokenizer

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Return total subword pieces divided by whitespace words."""
    total_pieces = 0
    total_words = 0

    for text in texts:
        words = text.split()
        if not words:
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        total_pieces += len(encoded["input_ids"])
        total_words += len(words)

    if total_words == 0:
        return 0.0

    return total_pieces / total_words


def percentile(values, percentile=95):
    """Simple linear-interpolation percentile."""
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(values[lower])

    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def load_texts():
    """Load Arabic and English text slices from the Bayan dataset."""
    ar_texts = []
    en_texts = []

    with DATA.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = (row.get("text") or "").strip()
            lang = (row.get("lang") or "").strip().lower()

            if not text:
                continue

            if lang == "ar":
                ar_texts.append(text)
            elif lang == "en":
                en_texts.append(text)

    return ar_texts, en_texts


def audit_tokenizer(tokenizer, texts):
    """Measure fertility, p95 sequence length and Arabic UNK rate."""
    if not texts:
        return 0.0, 0.0, 0.0

    lengths = []
    unk_tokens = 0
    total_tokens = 0

    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        token_ids = encoded["input_ids"]
        lengths.append(len(token_ids))

        total_tokens += len(token_ids)

        if tokenizer.unk_token_id is not None:
            unk_tokens += sum(
                token_id == tokenizer.unk_token_id
                for token_id in token_ids
            )

    fert = fertility(tokenizer, texts)
    p95_len = percentile(lengths, 95)

    unk_rate = (
        unk_tokens / total_tokens
        if total_tokens
        else 0.0
    )

    return fert, p95_len, unk_rate


def main():
    ar_texts, en_texts = load_texts()

    print(f"Loaded {len(ar_texts)} Arabic texts and {len(en_texts)} English texts.")
    print()

    print(
        f"{'Tokenizer':<15}"
        f"{'AR fertility':>14}"
        f"{'EN fertility':>14}"
        f"{'AR p95 len':>14}"
        f"{'EN p95 len':>14}"
        f"{'AR UNK rate':>14}"
    )
    print("-" * 85)

    results = []

    for model_name, short_name in CANDIDATES.items():
        print(f"Loading {short_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        ar_fert, ar_p95, ar_unk = audit_tokenizer(
            tokenizer,
            ar_texts,
        )

        en_fert, en_p95, _ = audit_tokenizer(
            tokenizer,
            en_texts,
        )

        results.append(
            (
                short_name,
                ar_fert,
                en_fert,
                ar_p95,
                en_p95,
                ar_unk,
            )
        )

        print(
            f"{short_name:<15}"
            f"{ar_fert:>14.3f}"
            f"{en_fert:>14.3f}"
            f"{ar_p95:>14.1f}"
            f"{en_p95:>14.1f}"
            f"{ar_unk:>14.4f}"
        )

    print()
    print("Audit complete.")
    print("Use these measured values to fill BENCHMARKS.md.")


if __name__ == "__main__":
    main()