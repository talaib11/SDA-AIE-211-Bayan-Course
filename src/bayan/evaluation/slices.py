"""Lab 6: sliced evaluation report."""

from collections import defaultdict

import numpy as np


def sliced_report(
    y_true,
    y_pred,
    *,
    languages=None,
    dialects=None,
    lengths=None,
    min_slice_size=20,
):
    """
    Build an evaluation report across useful slices:
    language, dialect, class, and input length.

    Returns a dictionary containing accuracy and sample
    count for each slice. Small slices are explicitly flagged.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length"
        )

    n = len(y_true)

    if n == 0:
        raise ValueError(
            "evaluation data must not be empty"
        )

    def validate_optional(values, name):
        if values is None:
            return None

        values = np.asarray(values)

        if len(values) != n:
            raise ValueError(
                f"{name} must have the same length as y_true"
            )

        return values

    languages = validate_optional(
        languages,
        "languages",
    )

    dialects = validate_optional(
        dialects,
        "dialects",
    )

    lengths = validate_optional(
        lengths,
        "lengths",
    )

    correct = y_true == y_pred

    report = {
        "overall": {
            "n": int(n),
            "accuracy": float(np.mean(correct)),
            "small_slice": n < min_slice_size,
        },
        "language": {},
        "dialect": {},
        "class": {},
        "length": {},
    }

    def add_categorical_slices(
        values,
        destination,
    ):
        if values is None:
            return

        for value in np.unique(values):
            mask = values == value

            count = int(np.sum(mask))

            report[destination][str(value)] = {
                "n": count,
                "accuracy": float(
                    np.mean(correct[mask])
                ),
                "small_slice": (
                    count < min_slice_size
                ),
            }

    # Language slices
    add_categorical_slices(
        languages,
        "language",
    )

    # Dialect slices
    add_categorical_slices(
        dialects,
        "dialect",
    )

    # Class slices
    add_categorical_slices(
        y_true,
        "class",
    )

    # Length slices
    if lengths is not None:
        length_groups = defaultdict(list)

        for index, length in enumerate(lengths):
            length = float(length)

            if length <= 10:
                bucket = "short"
            elif length <= 30:
                bucket = "medium"
            else:
                bucket = "long"

            length_groups[bucket].append(
                index
            )

        for bucket in [
            "short",
            "medium",
            "long",
        ]:
            indices = length_groups.get(
                bucket,
                [],
            )

            if not indices:
                continue

            indices = np.asarray(
                indices,
                dtype=int,
            )

            count = len(indices)

            report["length"][bucket] = {
                "n": int(count),
                "accuracy": float(
                    np.mean(correct[indices])
                ),
                "small_slice": (
                    count < min_slice_size
                ),
            }

    return report