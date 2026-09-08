"""Lab 3A: TF-IDF + LinearSVC baseline."""

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main():
    # Load the supplied dataset.
    df = pd.read_csv("data/raw/bayan_feedback.csv")

    # Use the supplied deterministic split.
    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    # Classical NLP baseline.
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LinearSVC()),
        ]
    )

    # Train on the training split only.
    start = time.perf_counter()

    model.fit(
        train_df["text"],
        train_df["topic"],
    )

    train_time = time.perf_counter() - start

    # Evaluate.
    validation_pred = model.predict(validation_df["text"])
    test_pred = model.predict(test_df["text"])

    validation_f1 = f1_score(
        validation_df["topic"],
        validation_pred,
        average="macro",
    )

    test_f1 = f1_score(
        test_df["topic"],
        test_pred,
        average="macro",
    )

    print(f"Validation macro-F1: {validation_f1:.4f}")
    print(f"Frozen test macro-F1: {test_f1:.4f}")
    print(f"Train time: {train_time:.2f} seconds")


if __name__ == "__main__":
    main()