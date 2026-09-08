"""Lab 4: compare Arabic-centric checkpoints by All/Gulf/MSA slices."""

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )

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
        df["lang"] == "ar"
    ].copy()

    train_df = arabic_df[
        arabic_df["split"] == "train"
    ].copy()

    eval_df = arabic_df[
        arabic_df["split"] == "validation"
    ].copy()

    print("Arabic rows:", len(arabic_df))
    print("Train rows:", len(train_df))
    print("Validation rows:", len(eval_df))

    print(
        "Arabic dialect distribution:",
        arabic_df["dialect_region"]
        .value_counts()
        .to_dict(),
    )

    print(
        "Validation dialect distribution:",
        eval_df["dialect_region"]
        .value_counts()
        .to_dict(),
    )

    if len(train_df) == 0:
        raise ValueError(
            "No Arabic training examples were found."
        )

    if len(eval_df) == 0:
        raise ValueError(
            "No Arabic validation examples were found."
        )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    labels = sorted(
        train_df["topic"].unique()
    )

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    train_df["labels"] = (
        train_df["topic"].map(label2id)
    )

    eval_df["labels"] = (
        eval_df["topic"].map(label2id)
    )

    print(
        "Number of topic labels:",
        len(labels),
    )

    print(
        "Topics:",
        labels,
    )

    # ---------------------------------------------------------
    # Prepare dialect slice indices
    # ---------------------------------------------------------

    dialect_regions = (
        eval_df["dialect_region"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    gulf_indices = np.array(
        [
            i
            for i, region in enumerate(dialect_regions)
            if region.lower() == "gulf"
        ],
        dtype=int,
    )

    msa_indices = np.array(
        [
            i
            for i, region in enumerate(dialect_regions)
            if region.lower() == "msa"
        ],
        dtype=int,
    )

    print(
        "Validation Gulf examples:",
        len(gulf_indices),
    )

    print(
        "Validation MSA examples:",
        len(msa_indices),
    )

    if len(gulf_indices) == 0:
        raise ValueError(
            "No Gulf examples found in the validation split. "
            "Cannot compute Gulf macro-F1."
        )

    if len(msa_indices) == 0:
        raise ValueError(
            "No MSA examples found in the validation split. "
            "Cannot compute MSA macro-F1."
        )

    # ---------------------------------------------------------
    # Model bake-off
    # ---------------------------------------------------------

    results = []

    for model_name, checkpoint in MODELS.items():
        print()
        print("=" * 70)
        print("Model:", model_name)
        print("Checkpoint:", checkpoint)
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint
        )

        def make_dataset(frame):
            dataset = Dataset.from_pandas(
                frame[
                    [
                        "text",
                        "labels",
                    ]
                ],
                preserve_index=False,
            )

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=128,
                )

            dataset = dataset.map(
                tokenize,
                batched=True,
            )

            return dataset

        train_ds = make_dataset(
            train_df
        )

        eval_ds = make_dataset(
            eval_df
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                checkpoint,
                num_labels=len(labels),
                label2id=label2id,
                id2label=id2label,
            )
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        def compute_metrics(eval_pred):
            logits, gold = eval_pred

            predictions = np.argmax(
                logits,
                axis=-1,
            )

            return {
                "macro_f1": macro_f1(
                    gold,
                    predictions,
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
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # -----------------------------------------------------
        # Frozen validation predictions
        # -----------------------------------------------------

        prediction_output = trainer.predict(
            eval_ds
        )

        predictions = np.argmax(
            prediction_output.predictions,
            axis=-1,
        )

        gold = np.asarray(
            prediction_output.label_ids,
            dtype=int,
        )

        predictions = np.asarray(
            predictions,
            dtype=int,
        )

        # -----------------------------------------------------
        # All Arabic
        # -----------------------------------------------------

        all_f1 = macro_f1(
            gold,
            predictions,
        )

        # -----------------------------------------------------
        # Gulf slice
        # -----------------------------------------------------

        gulf_gold = gold[
            gulf_indices
        ]

        gulf_predictions = predictions[
            gulf_indices
        ]

        gulf_f1 = macro_f1(
            gulf_gold,
            gulf_predictions,
        )

        # -----------------------------------------------------
        # MSA slice
        # -----------------------------------------------------

        msa_gold = gold[
            msa_indices
        ]

        msa_predictions = predictions[
            msa_indices
        ]

        msa_f1 = macro_f1(
            msa_gold,
            msa_predictions,
        )

        row = {
            "model": model_name,
            "all_macro_f1": float(all_f1),
            "gulf_macro_f1": float(gulf_f1),
            "msa_macro_f1": float(msa_f1),
        }

        results.append(row)

        print()
        print(
            "All Arabic macro-F1:",
            f"{all_f1:.4f}",
        )

        print(
            "Gulf macro-F1:",
            f"{gulf_f1:.4f}",
        )

        print(
            "MSA macro-F1:",
            f"{msa_f1:.4f}",
        )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ARABIC MODEL BAKE-OFF")
    print("=" * 70)

    for row in results:
        print(
            f'{row["model"]}: '
            f'All={row["all_macro_f1"]:.4f} | '
            f'Gulf={row["gulf_macro_f1"]:.4f} | '
            f'MSA={row["msa_macro_f1"]:.4f}'
        )

    # ---------------------------------------------------------
    # Day-2 baseline
    # ---------------------------------------------------------

    baseline = next(
        row
        for row in results
        if row["model"] == "Day-2 XLM-R"
    )

    # Only Arabic-centric models are candidates for Lab 4 winner.
    arabic_models = [
        row
        for row in results
        if row["model"] != "Day-2 XLM-R"
    ]

    # Course contract: choose using Gulf slice evidence.
    winner = max(
        arabic_models,
        key=lambda row: row["gulf_macro_f1"],
    )

    gulf_delta = (
        winner["gulf_macro_f1"]
        - baseline["gulf_macro_f1"]
    )

    print()
    print("-" * 70)

    print(
        "Winner by Gulf slice:",
        winner["model"],
    )

    print(
        "Day-2 Gulf macro-F1:",
        f'{baseline["gulf_macro_f1"]:.4f}',
    )

    print(
        "Winner Gulf macro-F1:",
        f'{winner["gulf_macro_f1"]:.4f}',
    )

    print(
        "Gulf macro-F1 delta vs Day-2:",
        f"{gulf_delta:+.4f}",
    )

    if gulf_delta >= 0.04:
        print(
            "Lab 4 Gulf target: MET"
        )
    else:
        print(
            "Lab 4 Gulf target: NOT MET"
        )

    print("-" * 70)


if __name__ == "__main__":
    main()