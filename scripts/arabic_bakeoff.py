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
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    return parser.parse_args()


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
    )


def main():
    args = parse_args()

    df = pd.read_csv(DATA_PATH)

    # Arabic slice only.
    df = df[df["lang"] == "ar"].copy()

    # Use the supplied frozen splits.
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    labels = sorted(train_df["topic"].unique())

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    for frame in (train_df, valid_df, test_df):
        frame["labels"] = frame["topic"].map(label2id)

    results = []

    for model_name, checkpoint in MODELS.items():
        print()
        print("=" * 60)
        print("Model:", model_name)
        print("Checkpoint:", checkpoint)
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
        )

        def to_dataset(frame):
            ds = Dataset.from_pandas(
                frame[
                    [
                        "text",
                        "labels",
                        "dialect_region",
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

            return ds.map(
                tokenize,
                batched=True,
            )

        train_ds = to_dataset(train_df)
        valid_ds = to_dataset(valid_df)
        test_ds = to_dataset(test_df)

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        def compute_metrics(eval_pred):
            logits, labels_array = eval_pred

            predictions = np.argmax(
                logits,
                axis=-1,
            )

            return {
                "macro_f1": macro_f1(
                    labels_array,
                    predictions,
                )
            }

        training_args = TrainingArguments(
            output_dir=f"/content/artifacts/{model_name}",
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
            eval_dataset=valid_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        prediction_output = trainer.predict(test_ds)

        predictions = np.argmax(
            prediction_output.predictions,
            axis=-1,
        )

        gold = prediction_output.label_ids

        # All Arabic.
        all_f1 = macro_f1(
            gold,
            predictions,
        )

        dialects = test_df["dialect_region"].tolist()

        gulf_indices = [
            i
            for i, region in enumerate(dialects)
            if region == "Gulf"
        ]

        msa_indices = [
            i
            for i, region in enumerate(dialects)
            if region == "MSA"
        ]

        gulf_f1 = macro_f1(
            gold[gulf_indices],
            predictions[gulf_indices],
        )

        msa_f1 = macro_f1(
            gold[msa_indices],
            predictions[msa_indices],
        )

        results.append(
            {
                "model": model_name,
                "all_macro_f1": all_f1,
                "gulf_macro_f1": gulf_f1,
                "msa_macro_f1": msa_f1,
            }
        )

        print()
        print("All Arabic macro-F1:", all_f1)
        print("Gulf macro-F1:", gulf_f1)
        print("MSA macro-F1:", msa_f1)

    print()
    print("=" * 60)
    print("FINAL BAKE-OFF")
    print("=" * 60)

    for row in results:
        print(
            f'{row["model"]}: '
            f'All={row["all_macro_f1"]:.4f} | '
            f'Gulf={row["gulf_macro_f1"]:.4f} | '
            f'MSA={row["msa_macro_f1"]:.4f}'
        )

    winner = max(
        results,
        key=lambda row: row["gulf_macro_f1"],
    )

    print()
    print(
        "Winner by Gulf slice:",
        winner["model"],
    )


if __name__ == "__main__":
    main()