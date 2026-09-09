"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """
    Return:
        point estimate,
        lower bootstrap percentile bound,
        upper bootstrap percentile bound.
    """

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)

    point = float(np.mean(values))

    n = len(values)

    boot_means = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = values[indices]

        boot_means[i] = np.mean(
            sample
        )

    lo = float(
        np.quantile(
            boot_means,
            alpha / 2,
        )
    )

    hi = float(
        np.quantile(
            boot_means,
            1 - alpha / 2,
        )
    )

    return point, lo, hi


def paired_bootstrap_diff(
    a,
    b,
    *,
    n_boot=2000,
    seed=42,
    alpha=0.05,
):
    """
    Paired bootstrap for the mean difference a - b.

    Pairing is preserved by resampling the same indices
    for both arrays.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            "paired inputs must have the same length"
        )

    if len(a) == 0:
        raise ValueError(
            "paired inputs must not be empty"
        )

    if n_boot <= 0:
        raise ValueError(
            "n_boot must be positive"
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1"
        )

    rng = np.random.default_rng(seed)

    diffs = a - b

    delta = float(
        np.mean(diffs)
    )

    n = len(diffs)

    boot_deltas = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot_deltas[i] = np.mean(
            diffs[indices]
        )

    lo = float(
        np.quantile(
            boot_deltas,
            alpha / 2,
        )
    )

    hi = float(
        np.quantile(
            boot_deltas,
            1 - alpha / 2,
        )
    )

    return delta, lo, hi