"""Lab 4: compare Arabic-centric checkpoints on all/Gulf/MSA slices."""

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

    df = pd.read_csv(DATA_PATH)

    # Arabic slice only.
    arabic_df = df[df["lang"] == "ar"].copy()

    # The supplied Arabic data contains train + validation.
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
        "Validation dialects:",
        eval_df["dialect_region"].value_counts().to_dict(),
    )

    # Topic labels.
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

    train_df["labels"] = train_df[
        "topic"
    ].map(label2id)

    eval_df["labels"] = eval_df[
        "topic"
    ].map(label2id)

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

            return dataset.map(
                tokenize,
                batched=True,
            )

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

        prediction_output = trainer.predict(
            eval_ds
        )

        predictions = np.argmax(
            prediction_output.predictions,
            axis=-1,
        )

        gold = prediction_output.label_ids

        # All Arabic validation slice.
        all_f1 = macro_f1(
            gold,
            predictions,
        )

        dialect_regions = (
            eval_df["dialect_region"]
            .tolist()
        )

        gulf_indices = np.array(
            [
                i
                for i, region
                in enumerate(dialect_regions)
                if region == "Gulf"
            ]
        )

        msa_indices = np.array(
            [
                i
                for i, region
                in enumerate(dialect_regions)
                if region == "MSA"
            ]
        )

        gulf_f1 = macro_f1(
            gold[gulf_indices],
            predictions[gulf_indices],
        )

        msa_f1 = macro_f1(
            gold[msa_indices],
            predictions[msa_indices],
        )

        row = {
            "model": model_name,
            "all_macro_f1": all_f1,
            "gulf_macro_f1": gulf_f1,
            "msa_macro_f1": msa_f1,
        }

        results.append(row)

        print()
        print(
            "All Arabic macro-F1:",
            round(all_f1, 4),
        )
        print(
            "Gulf macro-F1:",
            round(gulf_f1, 4),
        )
        print(
            "MSA macro-F1:",
            round(msa_f1, 4),
        )

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

    # Day-2 baseline for Lab 4 target.
    baseline = next(
        row
        for row in results
        if row["model"] == "Day-2 XLM-R"
    )

    arabic_models = [
        row
        for row in results
        if row["model"] != "Day-2 XLM-R"
    ]

    winner = max(
        arabic_models,
        key=lambda row: row["gulf_macro_f1"],
    )

    gulf_delta = (
        winner["gulf_macro_f1"]
        - baseline["gulf_macro_f1"]
    )

    print()
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


if __name__ == "__main__":
    main()