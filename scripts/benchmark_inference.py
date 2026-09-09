"""Lab 7: honest CPU p50/p99 benchmark over production serving mix."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = Path("artifacts/topic_classifier")
MIX_PATH = Path("data/serving/bench_mix.npy")

THREADS = 4

os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(THREADS)


def benchmark(
    model,
    tokenizer,
    texts,
    *,
    max_length,
    dynamic_padding,
    warmup=2,
    runs=10,
):
    """
    Measure CPU inference latency.

    Returns p50 and p99 latency in milliseconds.
    """

    model = model.to("cpu")
    model.eval()

    texts = np.asarray(texts).astype(str)

    if len(texts) == 0:
        raise ValueError("Production benchmark mix is empty.")

    # Spread the measured examples across the full production mix.
    indices = np.linspace(
        0,
        len(texts) - 1,
        num=runs,
        dtype=int,
    )

    measured_texts = texts[indices]

    def run_one(text):
        encoded = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=(
                False
                if dynamic_padding
                else "max_length"
            ),
        )

        with torch.inference_mode():
            model(**encoded)

    # Warm-up is excluded from measurements.
    for i in range(warmup):
        run_one(
            measured_texts[
                i % len(measured_texts)
            ]
        )

    latencies = []

    for text in measured_texts:
        start = time.perf_counter()

        run_one(text)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

    return {
        "p50_ms": float(
            np.percentile(latencies, 50)
        ),
        "p99_ms": float(
            np.percentile(latencies, 99)
        ),
        "min_ms": float(min(latencies)),
        "max_ms": float(max(latencies)),
        "runs": len(latencies),
    }


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Classifier not found at {MODEL_DIR}. "
            "Restore or train the Lab 3 classifier first."
        )

    if not MIX_PATH.exists():
        raise FileNotFoundError(
            f"Production mix not found at {MIX_PATH}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_DIR)
    )

    texts = np.load(
        MIX_PATH,
        allow_pickle=True,
    ).astype(str)

    print("=" * 70)
    print("LAB 7 — CPU INFERENCE BENCHMARK")
    print("=" * 70)
    print("Device: CPU")
    print(f"OMP_NUM_THREADS: {THREADS}")
    print(
        f"Production mix examples: {len(texts)}"
    )
    print()

    # ---------------------------------------------------------
    # Rung 1 — FP32 baseline
    # Intentionally padded to 512.
    # This is expensive on CPU, so use a compact measured sample.
    # ---------------------------------------------------------

    print("Running FP32 512-padded baseline...")

    baseline = benchmark(
        model,
        tokenizer,
        texts,
        max_length=512,
        dynamic_padding=False,
        warmup=2,
        runs=10,
    )

    # ---------------------------------------------------------
    # Rung 2 — Free wins
    # Dynamic padding + shorter max length.
    # ---------------------------------------------------------

    print(
        "Running FP32 dynamic-padding / "
        "max_length=128..."
    )

    free_wins = benchmark(
        model,
        tokenizer,
        texts,
        max_length=128,
        dynamic_padding=True,
        warmup=5,
        runs=30,
    )

    speedup = (
        baseline["p99_ms"]
        / free_wins["p99_ms"]
    )

    print()
    print("Optimisation ladder")
    print("-" * 70)

    print(
        "FP32 PyTorch | "
        "max_length=512 padded | "
        f"p50={baseline['p50_ms']:.2f} ms | "
        f"p99={baseline['p99_ms']:.2f} ms | "
        f"runs={baseline['runs']}"
    )

    print(
        "FP32 PyTorch | "
        "dynamic padding / max_length=128 | "
        f"p50={free_wins['p50_ms']:.2f} ms | "
        f"p99={free_wins['p99_ms']:.2f} ms | "
        f"speed-up={speedup:.2f}x | "
        f"runs={free_wins['runs']}"
    )


if __name__ == "__main__":
    main()