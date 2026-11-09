"""Split-conformal order statistics for fixed, exchangeable calibration units.

These functions do not establish exchangeability. In particular, using the same
units for model selection and calibration, calibrating individual blocks of a
slice as if independent, or adapting a policy after inspecting the test set does
not acquire a coverage guarantee by calling an order-statistic helper.
"""

from __future__ import annotations

import math

import numpy as np


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return score rank ``ceil((n+1)*(1-alpha))``, or infinity if unavailable.

    A rank beyond the calibration sample has to be represented by an infinite
    bound. Clipping to the largest score silently undercovers at small ``n``.
    Scores may be real or positive infinity; NaN and negative infinity are
    invalid. ``alpha`` must be strictly between zero and one.
    """

    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("calibration scores must be a nonempty one-dimensional array")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("calibration scores must not contain NaN or negative infinity")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def error_ratios(errors: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Compute nonnegative error/bound scores without hiding zero-bound failures."""

    errors, bounds = np.broadcast_arrays(
        np.asarray(errors, dtype=float), np.asarray(bounds, dtype=float)
    )
    if np.any(~np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("errors must be finite and nonnegative")
    if np.any(~np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError("reference bounds must be finite and nonnegative")
    ratios = np.zeros(errors.shape, dtype=float)
    np.divide(errors, bounds, out=ratios, where=bounds > 0)
    ratios[(bounds == 0) & (errors > 0)] = np.inf
    return ratios


def scale_error_bounds(bounds: np.ndarray, scale: float) -> np.ndarray:
    """Scale finite nonnegative bounds; an unsupported scale always yields infinity.

    The explicit infinity branch avoids ``infinity * 0 = NaN`` and rejects all
    statistical acceptances when calibration cannot support the requested rank.
    """

    bounds = np.asarray(bounds, dtype=float)
    if np.any(~np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError("reference bounds must be finite and nonnegative")
    if np.isnan(scale) or scale < 0:
        raise ValueError("scale must be nonnegative and not NaN")
    if np.isposinf(scale):
        return np.full(bounds.shape, np.inf, dtype=float)
    return bounds * float(scale)
