"""Lab 4: per-model Arabic normalisation profiles."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    result = text

    # Remove Arabic diacritics when requested.
    if profile.dediacritize:
        result = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", result)

    if profile.name == "bayan_ar_v1":
        # Remove tatweel.
        result = result.replace("ـ", "")

        # Normalize Alef variants.
        result = re.sub(r"[إأآ]", "ا", result)

        # Normalize Hamza-on-Waw.
        result = result.replace("ؤ", "و")

        # Normalize Alef Maqsura.
        result = result.replace("ى", "ي")

        # Normalize Taa Marbuta.
        result = result.replace("ة", "ه")

    return result


def segment(text: str) -> list[str]:
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer

    disambig = MLEDisambiguator.pretrained()

    tokenizer = MorphologicalTokenizer(
        disambig,
        scheme="d3tok",
        split=True,
    )

    return tokenizer.tokenize([text])[0]