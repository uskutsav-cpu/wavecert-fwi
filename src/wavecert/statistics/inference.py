"""Uncertainty summaries that distinguish observations from independent units."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def _confidence(value: float) -> float:
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    return float(value)


def binomial_failure_summary(failures: np.ndarray, confidence: float = 0.95) -> dict:
    """Exact one-sided Clopper-Pearson failure upper limit for independent trials.

    Callers must supply independent experimental units. This interval is not
    valid for correlated slices, repair rounds, or repeated draws on one case.
    For zero failures it remains positive: ``1-(1-confidence)**(1/n)``.
    """

    confidence = _confidence(confidence)
    raw = np.asarray(failures)
    if raw.ndim != 1 or np.any(~np.isin(raw, [0, 1])):
        raise ValueError("failures must be a one-dimensional array of booleans or 0/1")
    count = int(np.sum(raw))
    n = int(raw.size)
    upper = 1.0 if n == 0 or count == n else float(beta.ppf(confidence, count + 1, n - count))
    return {
        "n_units": n,
        "failures": count,
        "observed_failure_rate": count / n if n else None,
        "confidence": confidence,
        "one_sided_failure_probability_upper": upper,
        "method": "exact one-sided Clopper-Pearson",
        "assumption": "independent Bernoulli experimental units; not correlated slices or draws",
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 20260912,
) -> dict:
    """Percentile interval for a slice-weighted mean, resampling whole parents.

    All observations in a selected parent are retained, including multiplicity
    when that parent is sampled twice. The estimand is the observation-weighted
    mean, not an equally weighted mean of parent means. Parent groups must be
    independently sampled; this bootstrap cannot repair a biased test sample.
    """

    confidence = _confidence(confidence)
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.ndim != 1 or values.shape != groups.shape or not values.size:
        raise ValueError("values and groups must be nonempty aligned one-dimensional arrays")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples < 100:
        raise ValueError("n_resamples must be an integer of at least 100")
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse)
    result = {
        "n_observations": int(values.size),
        "n_parent_groups": int(unique.size),
        "mean": float(np.mean(values)),
        "confidence": confidence,
        "interval": None,
        "method": "whole-parent percentile bootstrap of observation-weighted mean",
        "seed": int(seed),
        "n_resamples": int(n_resamples),
        "assumption": "independently sampled parent groups; slices within a parent may be dependent",
    }
    if unique.size < 2:
        result["status"] = "unavailable: fewer than two independent parent groups"
        return result
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_resamples))
    for i in range(int(n_resamples)):
        sampled = rng.integers(0, unique.size, size=unique.size)
        means[i] = np.sum(sums[sampled]) / np.sum(sizes[sampled])
    tail = 0.5 * (1.0 - confidence)
    result["interval"] = [float(v) for v in np.quantile(means, [tail, 1.0 - tail])]
    result["status"] = "descriptive interval conditional on the parent-sampling assumption"
    return result
