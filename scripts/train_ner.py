"""Lab 3B / Lab 4: fine-tune NER with optional segmented CoNLL data."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )

    parser.add_argument(
        "--data-path",
        default="data/models/bayan_ner.conll",
        help="Path to the CoNLL NER dataset.",
    )

    return parser.parse_args()


def read_conll(path):
    sentences = []
    labels = []

    words = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
                continue

            token, tag = line.split("\t")
            words.append(token)
            tags.append(tag)

    if words:
        sentences.append(words)
        labels.append(tags)

    return sentences, labels


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("NER data:", args.data_path)

    # Read either the original or segmented CoNLL dataset.
    sentences, tag_sequences = read_conll(args.data_path)

    # Build BIO label mappings.
    tag_names = sorted(
        {
            tag
            for sequence in tag_sequences
            for tag in sequence
        }
    )

    label2id = {
        label: idx
        for idx, label in enumerate(tag_names)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    numeric_labels = [
        [label2id[tag] for tag in sequence]
        for sequence in tag_sequences
    ]

    # Deterministic 70/20/10 split.
    n = len(sentences)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.90)

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "tokens": sentences[:train_end],
                    "ner_tags": numeric_labels[:train_end],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "tokens": sentences[
                        train_end:validation_end
                    ],
                    "ner_tags": numeric_labels[
                        train_end:validation_end
                    ],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "tokens": sentences[validation_end:],
                    "ner_tags": numeric_labels[validation_end:],
                }
            ),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        add_prefix_space=True,
    )

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        aligned_labels = []

        for batch_index, labels in enumerate(
            batch["ner_tags"]
        ):
            word_ids = tokenized.word_ids(
                batch_index=batch_index
            )

            aligned_labels.append(
                align_labels(
                    word_ids,
                    labels,
                )
            )

        tokenized["labels"] = aligned_labels

        return tokenized

    tokenized = dataset.map(
        tokenize_and_align,
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(tag_names),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        true_predictions = []
        true_labels = []

        for pred_row, label_row in zip(
            predictions,
            labels,
        ):
            prediction_tags = []
            gold_tags = []

            for pred_id, label_id in zip(
                pred_row,
                label_row,
            ):
                if label_id == -100:
                    continue

                prediction_tags.append(
                    id2label[int(pred_id)]
                )

                gold_tags.append(
                    id2label[int(label_id)]
                )

            true_predictions.append(
                prediction_tags
            )

            true_labels.append(
                gold_tags
            )

        entity_f1 = f1_score(
            true_labels,
            true_predictions,
        )

        report = classification_report(
            true_labels,
            true_predictions,
            output_dict=True,
            zero_division=0,
        )

        location_recall = (
            report.get("LOCATION", {})
            .get("recall", 0.0)
        )

        return {
            "entity_f1": entity_f1,
            "location_recall": location_recall,
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        seed=42,
        report_to="none",
    )

    # Compatible with the newer Transformers Trainer API.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_metrics = trainer.evaluate(
        tokenized["validation"]
    )

    print()
    print(
        "Validation entity-F1:",
        validation_metrics["eval_entity_f1"],
    )
    print(
        "Validation LOCATION recall:",
        validation_metrics["eval_location_recall"],
    )

    test_metrics = trainer.evaluate(
        tokenized["test"]
    )

    print()
    print(
        "Frozen test entity-F1:",
        test_metrics["eval_entity_f1"],
    )
    print(
        "Frozen test LOCATION recall:",
        test_metrics["eval_location_recall"],
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print()
    print(
        "Saved NER model to:",
        output_dir,
    )


if __name__ == "__main__":
    main()