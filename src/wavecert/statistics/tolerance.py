"""Established one-sided IID tolerance calibration, separate from marginal CP.

The scoring rule, training/model selection and experimental unit must be fixed
before independent IID calibration. This helper cannot establish those facts.
It supplies no object-conditional, OOD or adaptive-trajectory guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


def _probability(value: float, name: str) -> float:
    if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
            or not 0 < value < 1):
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return float(value)


def _log_rank_failure(n: int, rank: int, population_coverage: float) -> float:
    # Some numerical binomial SF implementations underflow internally even
    # before the mathematical tail is subnormal. Log-PMF summation protects
    # very small delta requests; ordinary requests keep the fast SF path.
    return float(logsumexp(binom.logpmf(
        np.arange(rank, n + 1), n, population_coverage,
    )))


def _qualifying_rank(n: int, rank: int, alpha: float, delta: float) -> bool:
    if delta < 1e-12:
        return _log_rank_failure(n, rank, 1 - alpha) <= math.log(delta)
    return bool(binom.sf(rank - 1, n, 1 - alpha) <= delta)


def pac_tolerance_rank(n: int, *, alpha: float, delta: float) -> int | None:
    """Return the smallest qualifying 1-based order rank, or ``None``.

    For IID scores and population target coverage ``1-alpha``, rank k must
    satisfy ``BinomCDF(k-1; n, 1-alpha) >= 1-delta``. The survival-function
    comparison avoids rounding ``1-delta`` to one for very small delta.
    ``None`` means no finite sample order statistic supports that request;
    silently clipping to rank n would claim an unsupported confidence level.
    """
    alpha, delta = _probability(alpha, "alpha"), _probability(delta, "delta")
    if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or n < 0):
        raise ValueError("n must be a nonnegative integer")
    n = int(n)
    if n == 0 or not _qualifying_rank(n, n, alpha, delta):
        return None
    lower, upper = 1, n
    while lower < upper:
        candidate = (lower + upper) // 2
        if _qualifying_rank(n, candidate, alpha, delta):
            upper = candidate
        else:
            lower = candidate + 1
    return lower


@dataclass(frozen=True)
class ToleranceCalibration:
    """Rank support and numerical threshold are distinct deployment facts."""

    n_calibration: int
    alpha: float
    delta: float
    rank: int | None
    threshold: float
    rank_failure_probability_upper: float | None
    rank_failure_log_probability_upper: float | None

    def to_dict(self) -> dict:
        """Strict-JSON-compatible metadata; infinity is explicit, never NaN."""
        return {
            "n_calibration": self.n_calibration,
            "alpha": self.alpha,
            "delta": self.delta,
            "target_population_coverage": 1 - self.alpha,
            "requested_calibration_confidence": 1 - self.delta,
            "rank": self.rank,
            "finite_rank_supported": self.rank is not None,
            "threshold": self.threshold if np.isfinite(self.threshold) else None,
            "finite_threshold": bool(np.isfinite(self.threshold)),
            "threshold_kind": (
                "finite" if np.isfinite(self.threshold) else
                "positive_infinity_from_selected_score" if self.rank is not None else
                "positive_infinity_from_unsupported_rank"
            ),
            "rank_failure_probability_upper": self.rank_failure_probability_upper,
            "rank_failure_log_probability_upper": self.rank_failure_log_probability_upper,
            "method": "one-sided nonparametric IID tolerance order statistic (PAC calibration)",
            "assumption": "IID calibration and future score units conditional on a frozen training/scoring/selection rule; calibration is independent of fitting and model selection.",
            "scope": "Population score coverage with probability at least 1-delta over calibration samples; not object-conditional, OOD, or adaptive-trajectory coverage.",
        }


def pac_tolerance_threshold(
    scores: np.ndarray, *, alpha: float, delta: float,
) -> ToleranceCalibration:
    """Compute an optional tolerance threshold; preserve the raw mathematical q.

    With probability at least ``1-delta`` over IID calibration samples, the
    population probability of ``S_new <= q`` is at least ``1-alpha``. Ties are
    conservative. Scores may be real or positive infinity; NaN and negative
    infinity are rejected. A supported rank can select an infinite score,
    distinct from having too few calibration units to support a finite rank.

    Zero stays zero here. Deployment may require a separately disclosed
    conservative substitution to preserve a strictly positive repair scale.
    Infinite thresholds must abstain/fall back; they must not be clipped to a
    finite maximum or passed off as a useful finite calibration result.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("scores must not contain NaN or negative infinity")
    rank = pac_tolerance_rank(values.size, alpha=alpha, delta=delta)
    threshold = float("inf") if rank is None else float(np.partition(values, rank - 1)[rank - 1])
    failure = log_failure = None
    if rank is not None:
        if delta < 1e-12:
            log_failure = _log_rank_failure(values.size, rank, 1 - alpha)
            # Do not turn a positive mathematical tail into a reported zero
            # risk solely because its float64 exponential underflows.
            failure = max(math.exp(log_failure), float(np.nextafter(0.0, 1.0)))
        else:
            failure = float(binom.sf(rank - 1, values.size, 1 - alpha))
            log_failure = math.log(failure)
    return ToleranceCalibration(
        n_calibration=int(values.size), alpha=float(alpha), delta=float(delta),
        rank=rank, threshold=threshold, rank_failure_probability_upper=failure,
        rank_failure_log_probability_upper=log_failure,
    )
