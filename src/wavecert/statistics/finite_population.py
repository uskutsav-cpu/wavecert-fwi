"""Design-based uncertainty for a uniform sample from one fixed finite batch."""

from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom


def finite_population_failure_summary(
    failures: np.ndarray, *, population_size: int, confidence: float = 0.95,
) -> dict:
    """Invert the hypergeometric CDF for a conservative one-sided count bound.

    For observed k failures among n units drawn uniformly without replacement
    from N fixed units, retain the largest feasible K with
    ``Pr(X <= k | N, K, n) >= 1-confidence``. This covers only the failure
    fraction of those N units, conditional on fixed outcomes and the stated
    sampling design. It does not assume IID sites or infer another environment.
    """
    values = np.asarray(failures)
    if (values.ndim != 1 or not values.size or np.any(~np.isin(values, [0, 1]))):
        raise ValueError("failures must be a nonempty one-dimensional boolean/0-1 array")
    if (isinstance(population_size, (bool, np.bool_))
            or not isinstance(population_size, (int, np.integer))
            or population_size < len(values)):
        raise ValueError("population_size must be an integer at least the sample size")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    n, count, size = len(values), int(np.sum(values)), int(population_size)
    lower, upper = count, size - n + count
    tail = 1 - float(confidence)
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if hypergeom.cdf(count, size, candidate, n) >= tail:
            lower = candidate
        else:
            upper = candidate - 1
    return {
        "population_size": size, "sample_size": n, "observed_failures": count,
        "observed_sample_failure_fraction": count / n, "confidence": float(confidence),
        "one_sided_failure_count_upper": lower,
        "one_sided_failure_fraction_upper": lower / size,
        "method": "conservative inversion of the exact hypergeometric CDF",
        "scope": "Only the named fixed batch; no generalization to other batches, sites, families, or environments.",
        "assumption": "Uniform sampling without replacement; every unit outcome fixed before sampling and evaluation choices frozen.",
        "simultaneous_across_groups": False,
    }
