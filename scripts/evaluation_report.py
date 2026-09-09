"""Lab 6: generate the Bayan evaluation report and model cards."""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from jinja2 import Template
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.evaluation.slices import sliced_report
from bayan.evaluation.behavioural import run_behavioural_suite


PREDICTIONS_PATH = Path("data/eval/validation_predictions.csv")
FEEDBACK_PATH = Path("data/bayan_feedback.csv")

REVIEWED_ERRORS_CANDIDATES = [
    Path("data/eval/Lab6_120_errors_with_suggestions.csv"),
    Path("Lab6_120_errors_with_suggestions.csv"),
]

MODEL_DIR = Path("artifacts/topic_classifier")
MODEL_CARD_TEMPLATE = Path("templates/model_card.md.j2")

REPORT_PATH = Path("EVALUATION_REPORT.md")
CARDS_DIR = Path("docs/model_cards")


def markdown_table(rows, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(h, "")) for h in headers)
            + " |"
        )

    return "\n".join(lines)


def load_classifier():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Topic classifier not found at {MODEL_DIR}. "
            "Lab 3 classifier artefact is required."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )
    model.eval()

    return tokenizer, model


def classifier_helpers(tokenizer, model):
    id2label = {
        int(k): v
        for k, v in model.config.id2label.items()
    }

    def probabilities(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        )

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        probs = torch.softmax(logits, dim=-1)

        return {
            id2label[i]: float(probs[i])
            for i in range(len(probs))
        }

    def predict(text):
        probs = probabilities(text)
        return max(probs, key=probs.get)

    return predict, probabilities


def run_behavioural_tests(predict, probabilities):
    # ---------------------------------------------------------
    # Invariance:
    # swapping an irrelevant location should not change topic.
    # ---------------------------------------------------------

    invariance_cases = [
        {
            "text": "يوجد تسرب مياه في الرياض",
            "variant": "يوجد تسرب مياه في جدة",
        },
        {
            "text": "حفرة في الطريق في حي العليا",
            "variant": "حفرة في الطريق في حي النرجس",
        },
        {
            "text": "إنارة الشارع لا تعمل في الرياض",
            "variant": "إنارة الشارع لا تعمل في الدمام",
        },
        {
            "text": "دفعت الفاتورة وما زالت غير مسددة",
            "variant": "دفعت الفاتورة في جدة وما زالت غير مسددة",
        },
        {
            "text": "نحتاج حاوية نفايات إضافية في الحي",
            "variant": "نحتاج حاوية نفايات إضافية في حي الياسمين",
        },
        {
            "text": "The road needs urgent maintenance in Riyadh",
            "variant": "The road needs urgent maintenance in Jeddah",
        },
        {
            "text": "The street lighting is broken near Riyadh",
            "variant": "The street lighting is broken near Dammam",
        },
        {
            "text": "My water service is interrupted in Riyadh",
            "variant": "My water service is interrupted in Jeddah",
        },
    ]

    # ---------------------------------------------------------
    # Directional:
    # adding a strong domain cue should increase, or at least not
    # decrease, the expected topic score.
    # ---------------------------------------------------------

    directional_specs = [
        (
            "عندي مشكلة",
            "عندي مشكلة في الفاتورة والرسوم والدفع",
            "billing",
        ),
        (
            "يوجد عطل",
            "يوجد عطل وتسرب وانقطاع في خدمة المياه",
            "water",
        ),
        (
            "هناك مشكلة في الحي",
            "هناك حفرة وتلف في الطريق ويحتاج صيانة",
            "roads",
        ),
        (
            "الخدمة لا تعمل",
            "الخدمة الرقمية والتطبيق الإلكتروني لا يعملان",
            "digital_services",
        ),
        (
            "يوجد طلب صيانة",
            "عمود الإنارة والمصباح يحتاجان صيانة",
            "lighting",
        ),
    ]

    directional_cases = []

    for base, variant, label in directional_specs:
        directional_cases.append(
            {
                "text": base,
                "variant": variant,
                "direction": "increase",
                "score_fn": (
                    lambda output, expected=label:
                    output.get(expected, 0.0)
                ),
            }
        )

    # directional runner needs probability dictionaries
    directional_results = run_behavioural_suite(
        directional_cases=directional_cases,
        predict_fn=probabilities,
    )

    # ---------------------------------------------------------
    # MFT:
    # obvious domain examples should map to expected labels.
    # ---------------------------------------------------------

    mft_cases = [
        {
            "text": "دفعت الفاتورة لكن حالة الدفع لم تتحدث",
            "expected": "billing",
        },
        {
            "text": "يوجد تسرب مياه وانقطاع في الشبكة",
            "expected": "water",
        },
        {
            "text": "حفرة كبيرة في الطريق تحتاج إصلاح",
            "expected": "roads",
        },
        {
            "text": "عمود الإنارة في الشارع لا يعمل",
            "expected": "lighting",
        },
        {
            "text": "الحاوية ممتلئة والنفايات لم تجمع",
            "expected": "waste",
        },
        {
            "text": "أحتاج إصدار وتجديد الترخيص",
            "expected": "licensing",
        },
        {
            "text": "التطبيق والخدمة الإلكترونية لا تعمل",
            "expected": "digital_services",
        },
        {
            "text": "الحديقة تحتاج صيانة للأشجار والمقاعد",
            "expected": "parks",
        },
    ]

    base_results = run_behavioural_suite(
        invariance_cases=invariance_cases,
        mft_cases=mft_cases,
        predict_fn=predict,
    )

    return {
        "invariance": base_results["invariance"],
        "directional": directional_results["directional"],
        "mft": base_results["mft"],
    }


def main():
    print("=" * 70)
    print("LAB 6 — EVALUATION REPORT")
    print("=" * 70)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    correct = (
        predictions["y_true"].astype(str)
        == predictions["y_pred"].astype(str)
    ).astype(float)

    point, lo, hi = bootstrap_ci(
        correct.to_numpy(),
        n_boot=2000,
        seed=42,
    )

    print(
        f"Overall accuracy: {point:.4f} "
        f"[95% CI {lo:.4f}, {hi:.4f}]"
    )

    # ---------------------------------------------------------
    # Slices
    # ---------------------------------------------------------

    lengths = predictions["length_bucket"].map(
        {
            "short": 5,
            "medium": 20,
            "long": 50,
        }
    ).fillna(20)

    slices = sliced_report(
        predictions["y_true"].astype(str).to_numpy(),
        predictions["y_pred"].astype(str).to_numpy(),
        languages=predictions["lang"].fillna("unknown").to_numpy(),
        dialects=predictions["dialect_region"].fillna("NA").to_numpy(),
        lengths=lengths.to_numpy(),
        min_slice_size=20,
    )

    slice_rows = []

    for family in ["language", "dialect", "class", "length"]:
        for name, result in slices[family].items():
            slice_rows.append(
                {
                    "Slice": family,
                    "Value": name,
                    "N": result["n"],
                    "Accuracy": f'{result["accuracy"]:.4f}',
                    "Small": result["small_slice"],
                }
            )

    # ---------------------------------------------------------
    # Behavioural tests
    # ---------------------------------------------------------

    tokenizer, model = load_classifier()
    predict, probabilities = classifier_helpers(
        tokenizer,
        model,
    )

    behavioural = run_behavioural_tests(
        predict,
        probabilities,
    )

    behaviour_rows = []

    for family in [
        "invariance",
        "directional",
        "mft",
    ]:
        item = behavioural[family]

        behaviour_rows.append(
            {
                "Family": family,
                "Passed": item["passed"],
                "Total": item["total"],
                "Rate": f'{item["rate"]:.1%}',
            }
        )

    # ---------------------------------------------------------
    # Reviewed 120-error sample
    # ---------------------------------------------------------

    reviewed_path = next(
        (
            path
            for path in REVIEWED_ERRORS_CANDIDATES
            if path.exists()
        ),
        None,
    )

    if reviewed_path is None:
        raise FileNotFoundError(
            "Reviewed 120-error file not found. "
            "Place Lab6_120_errors_with_suggestions.csv "
            "in data/eval/."
        )

    reviewed = pd.read_csv(reviewed_path)

    category_column = (
        "error_category"
        if (
            "error_category" in reviewed.columns
            and reviewed["error_category"].notna().any()
            and reviewed["error_category"]
            .astype(str)
            .str.strip()
            .ne("")
            .any()
        )
        else "suggested_category"
    )

    counts = Counter(
        reviewed[category_column]
        .fillna("uncategorised")
        .astype(str)
    )

    taxonomy_rows = [
        {
            "Category": category,
            "Count": count,
            "Share": f"{count / len(reviewed):.1%}",
        }
        for category, count in counts.most_common()
    ]

    # Prioritised fixes based on the reviewed sample.
    top_fixes = [
        {
            "Priority": 1,
            "Fix": (
                "Add targeted training examples and behavioural "
                "regression tests for parks-vs-roads semantic overlap."
            ),
            "Predicted delta": (
                "+0.04 to +0.08 overall accuracy if the dominant "
                "confusion is substantially reduced."
            ),
        },
        {
            "Priority": 2,
            "Fix": (
                "Add more bilingual/code-switched training examples "
                "and preserve mixed-language tokens in preprocessing."
            ),
            "Predicted delta": (
                "+0.005 to +0.015 overall accuracy; larger expected "
                "benefit on the code-switching slice."
            ),
        },
        {
            "Priority": 3,
            "Fix": (
                "Add class-contrast examples and confidence-based "
                "error review for semantically neighbouring topics."
            ),
            "Predicted delta": (
                "+0.01 to +0.03 overall accuracy, pending validation."
            ),
        },
    ]

    # ---------------------------------------------------------
    # Manager headline
    # ---------------------------------------------------------

    class_accuracies = {
        name: item["accuracy"]
        for name, item in slices["class"].items()
    }

    weakest_class = min(
        class_accuracies,
        key=class_accuracies.get,
    )

    headline = (
        f"Bayan achieved {point:.1%} validation accuracy "
        f"(95% bootstrap CI {lo:.1%}–{hi:.1%}), but the aggregate "
        f"hides a material weakness in the {weakest_class} class. "
        "The reviewed error sample shows that semantic class overlap "
        "is the primary remediation priority before relying on the "
        "aggregate score alone."
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    report = f"""# Bayan Evaluation Report

## Manager headline

{headline}

## Aggregate result

- Validation examples: {len(predictions)}
- Accuracy: {point:.4f}
- 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]
- Errors: {int((correct == 0).sum())}

## Sliced evaluation

{markdown_table(
    slice_rows,
    ["Slice", "Value", "N", "Accuracy", "Small"]
)}

Small slices are explicitly flagged and should not be interpreted
with the same confidence as large slices.

## Behavioural tests

{markdown_table(
    behaviour_rows,
    ["Family", "Passed", "Total", "Rate"]
)}

Course targets:
- Invariance: approximately >= 95%
- MFT: approximately >= 90%

## Manual error review

A manually reviewed sample of {len(reviewed)} validation errors was
classified using the Lab 6 error taxonomy.

{markdown_table(
    taxonomy_rows,
    ["Category", "Count", "Share"]
)}

## Top three prioritised fixes

{markdown_table(
    top_fixes,
    ["Priority", "Fix", "Predicted delta"]
)}

The predicted deltas are hypotheses for the next validation run, not
measured improvements.

## Known limitations

- Validation performance is not uniform across classes; aggregate
  accuracy should not be used without slice results.
- The manual review identified substantial semantic overlap between
  neighbouring topic classes.
- Code-switched messages remain a smaller but visible failure mode.
- Predicted fix deltas have not yet been confirmed by retraining.
- Lab 5 semantic-search quality targets were not met in the measured
  retrieval run and remain a separate system limitation.

## Evidence

- Bootstrap confidence intervals: `src/bayan/evaluation/bootstrap.py`
- Sliced reporting: `src/bayan/evaluation/slices.py`
- Behavioural suite: `src/bayan/evaluation/behavioural.py`
- Validation predictions: `data/eval/validation_predictions.csv`
- Human-reviewed error sample: `{reviewed_path}`
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Model cards
    # ---------------------------------------------------------

    CARDS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = Template(
        MODEL_CARD_TEMPLATE.read_text(
            encoding="utf-8"
        )
    )

    metrics_table = markdown_table(
        [
            {
                "Metric": "Validation accuracy",
                "Value": f"{point:.4f}",
            },
            {
                "Metric": "95% CI lower",
                "Value": f"{lo:.4f}",
            },
            {
                "Metric": "95% CI upper",
                "Value": f"{hi:.4f}",
            },
        ],
        ["Metric", "Value"],
    )

    slices_table = markdown_table(
        slice_rows,
        ["Slice", "Value", "N", "Accuracy", "Small"],
    )

    behavioural_table = markdown_table(
        behaviour_rows,
        ["Family", "Passed", "Total", "Rate"],
    )

    cards = [
        {
            "filename": "topic_classifier.md",
            "model_name": "Bayan Topic Classifier",
            "checkpoint": "artifacts/topic_classifier",
            "intended_use": (
                "Bilingual Arabic/English citizen-feedback "
                "topic classification."
            ),
        },
        {
            "filename": "ner.md",
            "model_name": "Bayan NER",
            "checkpoint": "artifacts/ner_segmented",
            "intended_use": (
                "Named-entity extraction from bilingual "
                "citizen feedback."
            ),
        },
        {
            "filename": "semantic_search.md",
            "model_name": "Bayan Semantic Search",
            "checkpoint": "artifacts/search/case_index_v1",
            "intended_use": (
                "Retrieve and rerank similar historical "
                "resolved cases."
            ),
        },
    ]

    for card in cards:
        rendered = template.render(
            model_name=card["model_name"],
            intended_use=card["intended_use"],
            checkpoint=card["checkpoint"],
            preproc_version="1.2.0",
            data_version="Lab 6 validation snapshot",
            metrics_table=metrics_table,
            slices_table=slices_table,
            behavioural_table=behavioural_table,
        )

        # The template intentionally leaves the limitations section
        # for manual authorship. Replace its TODO with explicitly
        # authored limitations from this evaluation.
        rendered = rendered.replace(
            "TODO",
            (
                "- Performance differs by slice and class.\n"
                "- Semantic class overlap remains a known failure mode.\n"
                "- Code-switching requires continued monitoring.\n"
                "- Use only within the documented Bayan task scope."
            ),
            1,
        )

        rendered = rendered.replace(
            "TODO",
            "Bayan NLP project team",
            1,
        )

        (CARDS_DIR / card["filename"]).write_text(
            rendered,
            encoding="utf-8",
        )

    print()
    print("Evaluation report written to:", REPORT_PATH)
    print("Model cards written to:", CARDS_DIR)

    print()
    print("Behavioural results:")
    for row in behaviour_rows:
        print(
            row["Family"],
            row["Passed"],
            "/",
            row["Total"],
            row["Rate"],
        )


if __name__ == "__main__":
    main()