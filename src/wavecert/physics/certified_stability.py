"""Analytical stability bounds for the five-point Dirichlet reference system.

These bounds require the exact constant-spacing stiffness matrix implemented
by Helmholtz2D. They are not applicable to Devito, PML operators, variable
spacing, or another boundary condition without a new derivation. No linear
solve, matrix factorization, learned prediction, or SVD is used.

The inequalities are mathematical discrete-system bounds. Their evaluation
uses ordinary floating point with a small downward safety allowance, not
interval arithmetic or a formally verified roundoff bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class AnalyticalStabilityBound:
    beta_lower_bound: float
    method: str
    reference_slowness: float | None
    reference_damping: float | None
    reference_singular_gap: float
    perturbation_norm: float
    floating_point_allowance: float

    @property
    def available(self) -> bool:
        """Zero means these sufficient tests did not establish stability."""
        return bool(self.beta_lower_bound > 0.0)

    def to_dict(self) -> dict:
        return {**asdict(self), "available": self.available}


@lru_cache(maxsize=4)
def _laplacian_eigenvalues(shape: tuple[int, int], spacing: float) -> np.ndarray:
    nz, nx = shape
    z = np.sin(np.pi * np.arange(1, nz + 1) / (2 * (nz + 1))) ** 2
    x = np.sin(np.pi * np.arange(1, nx + 1) / (2 * (nx + 1))) ** 2
    values = (4.0 / spacing**2) * (z[:, None] + x[None, :])
    values.setflags(write=False)
    return values.reshape(-1)


def dirichlet_stability_lower_bound(
    *,
    shape: tuple[int, int],
    spacing: float,
    squared_slowness: np.ndarray,
    damping: np.ndarray,
    frequency_hz: float,
    center_points: int = 5,
) -> AnalyticalStabilityBound:
    r"""Lower-bound sigma_min(K - omega² diag(m) + i omega diag(d)).

    For a constant reference pair (m0, d0),

      B = K - omega² m0 I + i omega d0 I

    is normal and its singular values are

      sqrt((lambda_j(K) - omega² m0)² + omega² d0²).

    The diagonal perturbation E=A-B has spectral norm

      max_i sqrt(omega⁴ (m_i-m0)² + omega² (d_i-d0)²).

    The singular-value perturbation inequality therefore gives

      sigma_min(A) >= sigma_min(B) - ||E||_2.

    Every center gives a valid lower bound; maximizing over a fixed small
    grid preserves validity. Positive/negative real-part coercivity provides
    two further candidates. A non-positive result is returned as zero,
    never replaced by a guessed positive stability constant.

    Complexity is O(center_points² * number_of_grid_points), with O(n)
    storage. Increasing center_points can improve tightness but has no
    effect on the assumptions or coverage statement.
    """
    if len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
        raise ValueError("shape must contain two positive integers")
    shape = (int(shape[0]), int(shape[1]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if int(center_points) != center_points or center_points < 2:
        raise ValueError("center_points must be an integer of at least two")
    model = np.asarray(squared_slowness, dtype=float).reshape(-1)
    absorb = np.asarray(damping, dtype=float).reshape(-1)
    if model.size != np.prod(shape) or absorb.size != model.size:
        raise ValueError("model and damping must match the grid shape")
    if not np.all(np.isfinite(model)) or np.any(model <= 0):
        raise ValueError("squared slowness must be finite and positive")
    if not np.all(np.isfinite(absorb)) or np.any(absorb < 0):
        raise ValueError("damping must be finite and non-negative")

    omega = 2 * np.pi * float(frequency_hz)
    omega2 = omega**2
    eigenvalues = _laplacian_eigenvalues(shape, float(spacing))
    scale = float(np.max(eigenvalues) + omega2 * np.max(model) + omega * np.max(absorb))
    allowance = float(64 * np.finfo(float).eps * scale)
    best = AnalyticalStabilityBound(0.0, "unavailable", None, None, 0.0, 0.0, allowance)

    positive = float(np.min(eigenvalues) - omega2 * np.max(model))
    negative = float(omega2 * np.min(model) - np.max(eigenvalues))
    for method, gap in (("positive-real-part", positive), ("negative-real-part", negative)):
        lower = max(0.0, gap - allowance)
        if lower > best.beta_lower_bound:
            best = AnalyticalStabilityBound(lower, method, None, None, gap, 0.0, allowance)

    for model_center in np.linspace(np.min(model), np.max(model), int(center_points)):
        real_distance = eigenvalues - omega2 * model_center
        real_perturbation = omega2 * (model - model_center)
        for damping_center in np.linspace(np.min(absorb), np.max(absorb), int(center_points)):
            gap = float(np.min(np.hypot(real_distance, omega * damping_center)))
            perturbation = float(
                np.max(np.hypot(real_perturbation, omega * (absorb - damping_center)))
            )
            lower = max(0.0, gap - perturbation - allowance)
            if lower > best.beta_lower_bound:
                best = AnalyticalStabilityBound(
                    lower, "normal-reference-perturbation", float(model_center),
                    float(damping_center), gap, perturbation, allowance,
                )
    return best
