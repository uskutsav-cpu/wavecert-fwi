from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from wavecert.physics.helmholtz import Array, Helmholtz2D


@dataclass(frozen=True)
class DotProductResult:
    lhs: float
    rhs: float
    relative_error: float


@dataclass(frozen=True)
class GradientFDResult:
    adjoint_directional: float
    finite_difference: float
    relative_error: float
    epsilon: float


@dataclass(frozen=True)
class TaylorResult:
    epsilons: tuple[float, ...]
    remainders: tuple[float, ...]
    fitted_slope: float


@dataclass(frozen=True)
class VerificationReport:
    dot_tests: tuple[DotProductResult, ...]
    gradient_tests: tuple[GradientFDResult, ...]
    taylor_tests: tuple[TaylorResult, ...]

    def to_dict(self) -> dict:
        return {
            "dot_tests": [asdict(x) for x in self.dot_tests],
            "gradient_tests": [asdict(x) for x in self.gradient_tests],
            "taylor_tests": [asdict(x) for x in self.taylor_tests],
            "max_dot_relative_error": max(x.relative_error for x in self.dot_tests),
            "max_gradient_relative_error": max(x.relative_error for x in self.gradient_tests),
            "min_taylor_slope": min(x.fitted_slope for x in self.taylor_tests),
        }


def _relative_scalar_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-15)


def jvp_vjp_dot_test(
    physics: Helmholtz2D,
    m: Array,
    q: Array,
    receiver_indices: Iterable[int],
    frequency_hz: float,
    direction: Array,
    data_dual: Array,
) -> DotProductResult:
    jv = physics.data_jvp(m, q, receiver_indices, frequency_hz, direction)
    jty = physics.data_vjp(m, q, receiver_indices, frequency_hz, data_dual)
    lhs = float(np.real(np.vdot(np.asarray(data_dual), jv)))
    rhs = float(np.dot(np.asarray(direction, dtype=float).reshape(-1), jty))
    return DotProductResult(lhs, rhs, _relative_scalar_error(lhs, rhs))


def gradient_finite_difference_test(
    physics: Helmholtz2D,
    m: Array,
    q: Array,
    observed: Array,
    receiver_indices: Iterable[int],
    frequency_hz: float,
    direction: Array,
    *,
    epsilon: float = 1e-6,
) -> GradientFDResult:
    m = np.asarray(m, dtype=float).reshape(-1)
    direction = np.asarray(direction, dtype=float).reshape(-1)
    _, g, _, _ = physics.objective_and_gradient(
        m, q, observed, receiver_indices, frequency_hz
    )
    fp, _, _, _ = physics.objective_and_gradient(
        m + epsilon * direction, q, observed, receiver_indices, frequency_hz
    )
    fm, _, _, _ = physics.objective_and_gradient(
        m - epsilon * direction, q, observed, receiver_indices, frequency_hz
    )
    fd = float((fp - fm) / (2.0 * epsilon))
    ad = float(np.dot(g, direction))
    return GradientFDResult(ad, fd, _relative_scalar_error(ad, fd), float(epsilon))


def taylor_test(
    physics: Helmholtz2D,
    m: Array,
    q: Array,
    observed: Array,
    receiver_indices: Iterable[int],
    frequency_hz: float,
    direction: Array,
    *,
    epsilons: tuple[float, ...] = (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5),
) -> TaylorResult:
    m = np.asarray(m, dtype=float).reshape(-1)
    direction = np.asarray(direction, dtype=float).reshape(-1)
    f0, g, _, _ = physics.objective_and_gradient(
        m, q, observed, receiver_indices, frequency_hz
    )
    gd = float(np.dot(g, direction))
    rem: list[float] = []
    for eps in epsilons:
        fp, _, _, _ = physics.objective_and_gradient(
            m + eps * direction, q, observed, receiver_indices, frequency_hz
        )
        rem.append(abs(fp - f0 - eps * gd))

    e = np.asarray(epsilons, dtype=float)
    r = np.maximum(np.asarray(rem, dtype=float), np.finfo(float).tiny)
    # Fit all but any points that have fallen into a numerical round-off floor.
    mask = r > max(np.max(r) * 1e-10, 1e-18)
    if np.count_nonzero(mask) < 3:
        mask = np.ones_like(r, dtype=bool)
    slope = float(np.polyfit(np.log(e[mask]), np.log(r[mask]), 1)[0])
    return TaylorResult(tuple(map(float, e)), tuple(map(float, r)), slope)


def verify_reference_problem(
    physics: Helmholtz2D,
    m: Array,
    q: Array,
    observed: Array,
    receiver_indices: Iterable[int],
    frequency_hz: float,
    *,
    seed: int = 17,
    n_trials: int = 3,
) -> VerificationReport:
    rng = np.random.default_rng(seed)
    receiver_indices = tuple(receiver_indices)
    dots: list[DotProductResult] = []
    grads: list[GradientFDResult] = []
    taylors: list[TaylorResult] = []

    for _ in range(n_trials):
        v = rng.normal(size=physics.n)
        v /= max(np.linalg.norm(v), 1e-15)
        y = rng.normal(size=len(receiver_indices)) + 1j * rng.normal(size=len(receiver_indices))
        y /= max(np.linalg.norm(y), 1e-15)
        dots.append(jvp_vjp_dot_test(physics, m, q, receiver_indices, frequency_hz, v, y))
        grads.append(
            gradient_finite_difference_test(
                physics, m, q, observed, receiver_indices, frequency_hz, v
            )
        )
        taylors.append(
            taylor_test(physics, m, q, observed, receiver_indices, frequency_hz, v)
        )

    return VerificationReport(tuple(dots), tuple(grads), tuple(taylors))
