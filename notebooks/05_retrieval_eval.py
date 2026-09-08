"""Lab 5: labelled-query bilingual retrieval evaluation."""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from sklearn.metrics import f1_score

from bayan.search.service import CaseSearch
from bayan.search.index import normalize_text


PREFIX = "/content/artifacts/search/case_index_v1"
QUERY_PATH = "data/search/bayan_queries.jsonl"

K = 10
CANDIDATES = 50


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, case_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if case_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids)

    return len(
        relevant.intersection(retrieved)
    ) / len(relevant)


def evaluate_metrics(rows):
    recalls = [
        row["recall"]
        for row in rows
    ]

    reciprocal_ranks = [
        row["rr"]
        for row in rows
    ]

    return {
        "recall": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def main():
    queries = load_jsonl(
        QUERY_PATH
    )

    answerable = [
        row
        for row in queries
        if not row["no_answer"]
    ]

    no_answer = [
        row
        for row in queries
        if row["no_answer"]
    ]

    print("=" * 70)
    print("LAB 5 RETRIEVAL EVALUATION")
    print("=" * 70)

    print("Queries:", len(queries))
    print("Answerable:", len(answerable))
    print("No-answer:", len(no_answer))

    searcher = CaseSearch(
        PREFIX
    )

    # ---------------------------------------------------------
    # Stage 1: bi-encoder retrieval only
    # ---------------------------------------------------------

    print()
    print("Evaluating bi-encoder retrieval...")

    bi_rows = []
    bi_latencies = []

    for row in answerable:
        query = normalize_text(
            row["query"]
        )

        start = time.perf_counter()

        vector = searcher.encoder.encode(
            [query],
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(
            vector
        )

        scores, indices = searcher.index.search(
            vector,
            K,
        )

        latency = (
            time.perf_counter() - start
        ) * 1000.0

        bi_latencies.append(
            latency
        )

        retrieved_ids = []

        for index_id in indices[0]:
            if index_id < 0:
                continue

            retrieved_ids.append(
                searcher.metadata[
                    int(index_id)
                ]["case_id"]
            )

        bi_rows.append(
            {
                "query_id": row["query_id"],
                "lang": row["lang"],
                "recall": recall_at_k(
                    retrieved_ids,
                    row["relevant_case_ids"],
                ),
                "rr": reciprocal_rank(
                    retrieved_ids,
                    row["relevant_case_ids"],
                ),
            }
        )

    bi_metrics = evaluate_metrics(
        bi_rows
    )

    # ---------------------------------------------------------
    # Stage 2: retrieve 50 + cross-encoder rerank
    # ---------------------------------------------------------

    print("Evaluating cross-encoder reranking...")

    rerank_rows = []
    rerank_latencies = []

    top_scores_answerable = []

    for row in answerable:
        start = time.perf_counter()

        results = searcher.search(
            row["query"],
            k=K,
            candidates=CANDIDATES,
            min_score=-100.0,
        )

        latency = (
            time.perf_counter() - start
        ) * 1000.0

        rerank_latencies.append(
            latency
        )

        retrieved_ids = [
            item["case_id"]
            for item in results
        ]

        if results:
            top_scores_answerable.append(
                float(results[0]["score"])
            )

        rerank_rows.append(
            {
                "query_id": row["query_id"],
                "lang": row["lang"],
                "recall": recall_at_k(
                    retrieved_ids,
                    row["relevant_case_ids"],
                ),
                "rr": reciprocal_rank(
                    retrieved_ids,
                    row["relevant_case_ids"],
                ),
            }
        )

    rerank_metrics = evaluate_metrics(
        rerank_rows
    )

    # ---------------------------------------------------------
    # Language slices
    # ---------------------------------------------------------

    ar_rows = [
        row
        for row in rerank_rows
        if row["lang"] == "ar"
    ]

    en_rows = [
        row
        for row in rerank_rows
        if row["lang"] == "en"
    ]

    ar_metrics = evaluate_metrics(
        ar_rows
    )

    en_metrics = evaluate_metrics(
        en_rows
    )

    cross_lingual_gap = abs(
        ar_metrics["mrr"]
        - en_metrics["mrr"]
    )

    # ---------------------------------------------------------
    # No-answer score collection
    # ---------------------------------------------------------

    print("Evaluating no-answer slice...")

    no_answer_scores = []

    for row in no_answer:
        results = searcher.search(
            row["query"],
            k=1,
            candidates=CANDIDATES,
            min_score=-100.0,
        )

        if results:
            no_answer_scores.append(
                float(results[0]["score"])
            )
        else:
            no_answer_scores.append(
                float("-inf")
            )

    # ---------------------------------------------------------
    # Tune min_score using answerable + supplied 20 no-answer
    #
    # A query is considered answerable when top CE score >=
    # threshold. We select the threshold with the highest
    # correctness on the supplied labelled threshold slice.
    # ---------------------------------------------------------

    all_scores = (
        top_scores_answerable
        + no_answer_scores
    )

    finite_scores = [
        score
        for score in all_scores
        if np.isfinite(score)
    ]

    if not finite_scores:
        raise RuntimeError(
            "No finite cross-encoder scores were produced."
        )

    candidates = sorted(
        set(finite_scores)
    )

    epsilon = 1e-6

    thresholds = (
        [candidates[0] - epsilon]
        + candidates
        + [candidates[-1] + epsilon]
    )

    best_threshold = None
    best_correct = -1

    for threshold in thresholds:
        correct = 0

        for score in top_scores_answerable:
            if score >= threshold:
                correct += 1

        for score in no_answer_scores:
            if score < threshold:
                correct += 1

        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    no_answer_correct = sum(
        score < best_threshold
        for score in no_answer_scores
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL LAB 5 RESULTS")
    print("=" * 70)

    print(
        "Recall@10 without reranking:",
        f'{bi_metrics["recall"]:.4f}',
    )

    print(
        "MRR@10 without reranking:",
        f'{bi_metrics["mrr"]:.4f}',
    )

    print(
        "Recall@10 with reranking:",
        f'{rerank_metrics["recall"]:.4f}',
    )

    print(
        "MRR@10 with reranking:",
        f'{rerank_metrics["mrr"]:.4f}',
    )

    print()
    print(
        "Arabic reranked Recall@10:",
        f'{ar_metrics["recall"]:.4f}',
    )

    print(
        "Arabic reranked MRR@10:",
        f'{ar_metrics["mrr"]:.4f}',
    )

    print(
        "English reranked Recall@10:",
        f'{en_metrics["recall"]:.4f}',
    )

    print(
        "English reranked MRR@10:",
        f'{en_metrics["mrr"]:.4f}',
    )

    print(
        "Cross-lingual MRR gap:",
        f"{cross_lingual_gap:.4f}",
    )

    print()
    print(
        "Selected min_score:",
        f"{best_threshold:.6f}",
    )

    print(
        "No-answer correctness:",
        f"{no_answer_correct}/{len(no_answer)}",
    )

    print(
        "Threshold overall correctness:",
        f"{best_correct}/{len(top_scores_answerable) + len(no_answer_scores)}",
    )

    print()
    print(
        "Bi-encoder mean latency (ms/query):",
        f"{np.mean(bi_latencies):.2f}",
    )

    print(
        "Two-stage mean latency (ms/query):",
        f"{np.mean(rerank_latencies):.2f}",
    )

    print()
    print(
        "Recall target >= 0.80:",
        "MET"
        if rerank_metrics["recall"] >= 0.80
        else "NOT MET",
    )

    print(
        "MRR target >= 0.70:",
        "MET"
        if rerank_metrics["mrr"] >= 0.70
        else "NOT MET",
    )

    print(
        "No-answer target >= 17/20:",
        "MET"
        if no_answer_correct >= 17
        else "NOT MET",
    )


if __name__ == "__main__":
    main()