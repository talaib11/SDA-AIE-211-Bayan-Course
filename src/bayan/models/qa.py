"""Lab 3 starter: extractive QA post-processing."""


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    start_indices = sorted(
        range(len(start_logits)),
        key=lambda i: start_logits[i],
        reverse=True,
    )[:top_k]

    end_indices = sorted(
        range(len(end_logits)),
        key=lambda i: end_logits[i],
        reverse=True,
    )[:top_k]

    best_score = float("-inf")
    best = None

    for start_idx in start_indices:
        for end_idx in end_indices:

            # Skip special/question tokens with no context offset.
            if offsets[start_idx] is None or offsets[end_idx] is None:
                continue

            # End cannot come before start.
            if end_idx < start_idx:
                continue

            # Reject answers that are too long.
            if end_idx - start_idx + 1 > max_answer_len:
                continue

            score = (
                float(start_logits[start_idx])
                + float(end_logits[end_idx])
            )

            if score > best_score:
                best_score = score
                best = {
                    "answer": (
                        offsets[start_idx][0],
                        offsets[end_idx][1],
                    ),
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "score": score,
                }

    # Honest no-answer decision.
    if best is None:
        return {
            "answer": None,
            "score": float(null_score),
        }

    if float(null_score) - best_score > float(null_threshold):
        return {
            "answer": None,
            "score": float(null_score),
        }

    return best