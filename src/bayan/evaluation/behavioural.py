"""Lab 6: behavioural test generators and runners."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Any


def _safe_rate(passed: int, total: int) -> float:
    if total == 0:
        return 0.0
    return passed / total


def run_behavioural_suite(
    *,
    invariance_cases: Iterable[Mapping[str, Any]] | None = None,
    directional_cases: Iterable[Mapping[str, Any]] | None = None,
    mft_cases: Iterable[Mapping[str, Any]] | None = None,
    predict_fn: Callable[[str], Any] | None = None,
):
    """
    Run invariance, directional, and minimum-functionality tests.

    Expected case formats

    invariance:
        {
            "text": "...",
            "variant": "...",
            "expected_same": True
        }

    directional:
        {
            "text": "...",
            "variant": "...",
            "direction": "decrease" | "increase",
            "score_fn": callable | None
        }

    MFT:
        {
            "text": "...",
            "expected": <label/value>
        }

    Returns a dictionary with passed/total/rate for each family
    plus an overall summary.
    """

    if predict_fn is None:
        raise ValueError("predict_fn is required")

    invariance_cases = list(invariance_cases or [])
    directional_cases = list(directional_cases or [])
    mft_cases = list(mft_cases or [])

    results = {
        "invariance": {
            "passed": 0,
            "total": 0,
            "rate": 0.0,
            "details": [],
        },
        "directional": {
            "passed": 0,
            "total": 0,
            "rate": 0.0,
            "details": [],
        },
        "mft": {
            "passed": 0,
            "total": 0,
            "rate": 0.0,
            "details": [],
        },
    }

    # ---------------------------------------------------------
    # Invariance tests
    # Irrelevant transformations should preserve prediction.
    # Example: location swap should not change topic.
    # ---------------------------------------------------------

    for case in invariance_cases:
        text = case["text"]
        variant = case["variant"]

        base_pred = predict_fn(text)
        variant_pred = predict_fn(variant)

        expected_same = case.get(
            "expected_same",
            True,
        )

        passed = (
            base_pred == variant_pred
            if expected_same
            else base_pred != variant_pred
        )

        results["invariance"]["total"] += 1
        results["invariance"]["passed"] += int(passed)

        results["invariance"]["details"].append(
            {
                "text": text,
                "variant": variant,
                "base_prediction": base_pred,
                "variant_prediction": variant_pred,
                "passed": bool(passed),
            }
        )

    # ---------------------------------------------------------
    # Directional tests
    # A controlled perturbation should move a score in the
    # expected direction.
    #
    # Example: adding negation should not improve sentiment.
    # ---------------------------------------------------------

    for case in directional_cases:
        text = case["text"]
        variant = case["variant"]

        base_output = predict_fn(text)
        variant_output = predict_fn(variant)

        score_fn = case.get("score_fn")

        if score_fn is not None:
            base_score = float(score_fn(base_output))
            variant_score = float(score_fn(variant_output))
        else:
            base_score = float(base_output)
            variant_score = float(variant_output)

        direction = case.get(
            "direction",
            "decrease",
        )

        if direction == "decrease":
            passed = variant_score <= base_score
        elif direction == "increase":
            passed = variant_score >= base_score
        else:
            raise ValueError(
                "direction must be 'increase' or 'decrease'"
            )

        results["directional"]["total"] += 1
        results["directional"]["passed"] += int(passed)

        results["directional"]["details"].append(
            {
                "text": text,
                "variant": variant,
                "base_score": base_score,
                "variant_score": variant_score,
                "direction": direction,
                "passed": bool(passed),
            }
        )

    # ---------------------------------------------------------
    # Minimum functionality tests
    # Encode explicit domain contracts.
    #
    # Example: registry terms must be recognised.
    # ---------------------------------------------------------

    for case in mft_cases:
        text = case["text"]
        expected = case["expected"]

        prediction = predict_fn(text)

        comparator = case.get("comparator")

        if comparator is None:
            passed = prediction == expected
        else:
            passed = bool(
                comparator(
                    prediction,
                    expected,
                )
            )

        results["mft"]["total"] += 1
        results["mft"]["passed"] += int(passed)

        results["mft"]["details"].append(
            {
                "text": text,
                "prediction": prediction,
                "expected": expected,
                "passed": bool(passed),
            }
        )

    for family in (
        "invariance",
        "directional",
        "mft",
    ):
        results[family]["rate"] = _safe_rate(
            results[family]["passed"],
            results[family]["total"],
        )

    total = sum(
        results[family]["total"]
        for family in (
            "invariance",
            "directional",
            "mft",
        )
    )

    passed = sum(
        results[family]["passed"]
        for family in (
            "invariance",
            "directional",
            "mft",
        )
    )

    results["overall"] = {
        "passed": passed,
        "total": total,
        "rate": _safe_rate(
            passed,
            total,
        ),
    }

    return results