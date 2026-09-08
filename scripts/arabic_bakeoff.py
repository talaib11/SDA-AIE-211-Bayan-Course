"""Lab 4: Arabic model bake-off on All, Gulf, and MSA slices."""

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATA_PATH = "data/raw/bayan_feedback.csv"

MODELS = {
    "Day-2 XLM-R": "xlm-roberta-base",
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # Load Arabic data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    arabic_df = df[
        df["lang"].astype(str).str.lower() == "ar"
    ].copy()

    original_train = arabic_df[
        arabic_df["split"] == "train"
    ].copy()

    if original_train.empty:
        raise ValueError("No Arabic training rows found.")

    # ---------------------------------------------------------
    # Create a fixed evaluation split containing Gulf + MSA
    #
    # The supplied validation split contains only MSA, so it
    # cannot be used for the required Gulf-slice comparison.
    # We therefore create a deterministic held-out split from
    # the supplied Arabic training data, stratified jointly by
    # dialect and topic.
    # ---------------------------------------------------------

    original_train["stratum"] = (
        original_train["dialect_region"].astype(str)
        + "__"
        + original_train["topic"].astype(str)
    )

    train_df, eval_df = train_test_split(
        original_train,
        test_size=0.20,
        random_state=42,
        stratify=original_train["stratum"],
    )

    train_df = train_df.copy()
    eval_df = eval_df.copy()

    print("=" * 70)
    print("ARABIC BAKE-OFF DATA")
    print("=" * 70)

    print("Arabic rows total:", len(arabic_df))
    print("Bake-off train rows:", len(train_df))
    print("Bake-off evaluation rows:", len(eval_df))

    print(
        "Evaluation dialect distribution:",
        eval_df["dialect_region"].value_counts().to_dict(),
    )

    print(
        "Evaluation topic distribution:",
        eval_df["topic"].value_counts().to_dict(),
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    labels = sorted(
        original_train["topic"].unique()
    )

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    train_df["labels"] = train_df["topic"].map(label2id)
    eval_df["labels"] = eval_df["topic"].map(label2id)

    print("Number of topics:", len(labels))
    print("Topics:", labels)

    # ---------------------------------------------------------
    # Slice indices
    # ---------------------------------------------------------

    dialects = (
        eval_df["dialect_region"]
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    gulf_indices = np.array(
        [
            i
            for i, dialect in enumerate(dialects)
            if dialect == "gulf"
        ],
        dtype=int,
    )

    msa_indices = np.array(
        [
            i
            for i, dialect in enumerate(dialects)
            if dialect == "msa"
        ],
        dtype=int,
    )

    print("Evaluation Gulf rows:", len(gulf_indices))
    print("Evaluation MSA rows:", len(msa_indices))

    if len(gulf_indices) == 0:
        raise ValueError("Evaluation split contains no Gulf rows.")

    if len(msa_indices) == 0:
        raise ValueError("Evaluation split contains no MSA rows.")

    # ---------------------------------------------------------
    # Run bake-off
    # ---------------------------------------------------------

    results = []

    for model_name, checkpoint in MODELS.items():

        print()
        print("=" * 70)
        print("MODEL:", model_name)
        print("CHECKPOINT:", checkpoint)
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint
        )

        def make_dataset(frame):
            ds = Dataset.from_pandas(
                frame[["text", "labels"]],
                preserve_index=False,
            )

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=128,
                )

            return ds.map(
                tokenize,
                batched=True,
            )

        train_ds = make_dataset(train_df)
        eval_ds = make_dataset(eval_df)

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )

        collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        def compute_metrics(eval_pred):
            logits, gold = eval_pred

            pred = np.argmax(
                logits,
                axis=-1,
            )

            return {
                "macro_f1": macro_f1(
                    gold,
                    pred,
                )
            }

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        training_args = TrainingArguments(
            output_dir=(
                f"/content/artifacts/"
                f"arabic_bakeoff_{safe_name}"
            ),
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="no",
            seed=42,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        output = trainer.predict(eval_ds)

        predictions = np.argmax(
            output.predictions,
            axis=-1,
        )

        predictions = np.asarray(
            predictions,
            dtype=int,
        )

        gold = np.asarray(
            output.label_ids,
            dtype=int,
        )

        # All Arabic evaluation data
        all_f1 = macro_f1(
            gold,
            predictions,
        )

        # Gulf
        gulf_f1 = macro_f1(
            gold[gulf_indices],
            predictions[gulf_indices],
        )

        # MSA
        msa_f1 = macro_f1(
            gold[msa_indices],
            predictions[msa_indices],
        )

        results.append(
            {
                "model": model_name,
                "all": float(all_f1),
                "gulf": float(gulf_f1),
                "msa": float(msa_f1),
            }
        )

        print()
        print("RESULTS:", model_name)
        print("All macro-F1 :", f"{all_f1:.4f}")
        print("Gulf macro-F1:", f"{gulf_f1:.4f}")
        print("MSA macro-F1 :", f"{msa_f1:.4f}")

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ARABIC MODEL BAKE-OFF")
    print("=" * 70)

    for result in results:
        print(
            f'{result["model"]}: '
            f'All={result["all"]:.4f} | '
            f'Gulf={result["gulf"]:.4f} | '
            f'MSA={result["msa"]:.4f}'
        )

    baseline = next(
        result
        for result in results
        if result["model"] == "Day-2 XLM-R"
    )

    arabic_models = [
        result
        for result in results
        if result["model"] != "Day-2 XLM-R"
    ]

    winner = max(
        arabic_models,
        key=lambda result: result["gulf"],
    )

    gulf_delta = (
        winner["gulf"]
        - baseline["gulf"]
    )

    print()
    print("-" * 70)

    print(
        "Winner by Gulf slice:",
        winner["model"],
    )

    print(
        "Day-2 Gulf macro-F1:",
        f'{baseline["gulf"]:.4f}',
    )

    print(
        "Winner Gulf macro-F1:",
        f'{winner["gulf"]:.4f}',
    )

    print(
        "Gulf macro-F1 delta vs Day-2:",
        f"{gulf_delta:+.4f}",
    )

    if gulf_delta >= 0.04:
        print("Lab 4 Gulf target: MET")
    else:
        print("Lab 4 Gulf target: NOT MET")

    print("-" * 70)


if __name__ == "__main__":
    main()