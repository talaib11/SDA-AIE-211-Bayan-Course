"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "FFN": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        name_lower = name.lower()

        if "embeddings" in name_lower:
            buckets["embeddings"] += count

        elif "attention" in name_lower:
            buckets["attention"] += count

        elif "intermediate" in name_lower or "output.dense" in name_lower:
            buckets["FFN"] += count

        elif "layernorm" in name_lower:
            buckets["norms"] += count

        elif "pooler" in name_lower:
            buckets["pooler"] += count

        else:
            buckets["other"] += count

    buckets["total"] = sum(buckets.values())

    return buckets


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt)
        print(audit(ckpt))
        print()