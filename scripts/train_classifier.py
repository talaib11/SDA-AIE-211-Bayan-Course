"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        )
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load grouped train/validation/test dataset.
    ds = build_topic_dataset()

    # 2. Build label mapping.
    labels = sorted(set(ds["train"]["topic"]))

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    # 3. Load tokenizer chosen in Lab 1.
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

        encoded["labels"] = [
            label2id[label]
            for label in batch["topic"]
        ]

        return encoded

    tokenized = ds.map(
        tokenize_batch,
        batched=True,
    )

    # 4. Load pretrained XLM-R classifier.
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # 5. Fine-tuning configuration.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Train.
    trainer.train()

    # 7. Evaluate validation.
    validation_metrics = trainer.evaluate(
        tokenized["validation"]
    )

    print(
        "Validation macro-F1:",
        validation_metrics["eval_macro_f1"],
    )

    # 8. Evaluate frozen test ONCE.
    test_metrics = trainer.evaluate(
        tokenized["test"]
    )

    print(
        "Frozen test macro-F1:",
        test_metrics["eval_macro_f1"],
    )

    # 9. Save re-runnable artefact.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Saved classifier to:", output_dir)


if __name__ == "__main__":
    main()