"""Exact discrete survey gradients using residual-coded receiver solves.

These controls only reassociate the ordinary least-squares adjoint gradient.
They introduce no source randomization, approximation, or neural model. The
state operator is linear, sources and common sampling are fixed in the model,
and the model derivative is the reference diagonal mass term ``-omega² dm``.
Ordinary floating-point solves are not formal roundoff-enclosed certificates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.physics.receiver_oracle import _inputs, _real_vector


@dataclass(frozen=True)
class ExactSurveyGradientResult:
    objective: float
    gradient: np.ndarray
    frequency_strategies: dict[float, str]
    expected_rhs: int


def _finite_result(objective, gradient):
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("exact survey objective or gradient is non-finite")
    return float(objective), np.asarray(gradient, dtype=float)


def _complex_matrix(value, shape, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix of shape {shape}")
    return result


def matrix_residual_coded_gradient(
    *, physics: Helmholtz2D, m: np.ndarray, source_matrix: np.ndarray,
    receiver_matrix: np.ndarray, observed_matrix: np.ndarray, frequency_hz: float,
) -> tuple[float, np.ndarray]:
    r"""Reference helper for arbitrary fixed complex Q and common linear P.

    For ``C=P A^-1``, ``R=C Q-D`` and ``V=A^-1(Q R^H)``, the reference
    squared-slowness gradient is ``omega² Re diag(V C)``. Constructing C
    requires r adjoint RHS, and V requires r forward RHS. This identity does
    not assume ``A.T == A`` or Hermitian symmetry. Q and P must be independent
    of m; it does not include source/receiver model derivatives or another
    parameterization's operator derivative.
    """
    if not isinstance(physics, Helmholtz2D):
        raise TypeError("this helper requires the Helmholtz2D reference operator derivative")
    model = physics.validate_model(_real_vector(m, physics.n, "model"))
    q, p = np.asarray(source_matrix), np.asarray(receiver_matrix)
    if q.ndim != 2 or q.shape[0] != physics.n or q.shape[1] < 1:
        raise ValueError("source_matrix must have shape (n_grid, n_sources) with at least one source")
    if p.ndim != 2 or p.shape[1] != physics.n or p.shape[0] < 1:
        raise ValueError("receiver_matrix must have shape (n_receivers, n_grid) with at least one receiver")
    q = _complex_matrix(q, q.shape, "source_matrix")
    p = _complex_matrix(p, p.shape, "receiver_matrix")
    observed = _complex_matrix(observed_matrix, (p.shape[0], q.shape[1]), "observed_matrix")
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive")
    c = physics._solve(model, p.conjugate().T, frequency, category="adjoint_rhs", adjoint=True).conjugate().T
    residual = c @ q - observed
    coded = physics._solve(model, q @ residual.conjugate().T, frequency, category="forward_rhs")
    # C is NOT conjugated again: the conjugation is already in R^H.
    gradient = (2*np.pi*frequency)**2 * np.real(np.sum(c.T * coded, axis=1))
    objective = 0.5 * float(np.vdot(residual, residual).real)
    return _finite_result(objective, gradient)


def _receiver_map(physics, model, frequency, unique):
    rhs = np.zeros((physics.n, len(unique)), dtype=np.complex128)
    rhs[np.asarray(unique), np.arange(len(unique))] = 1.0
    # These receiver solves construct an exact gradient, so count them as
    # adjoint RHS rather than hiding them under an uncharged setup category.
    return physics._solve(model, rhs, frequency, category="adjoint_rhs", adjoint=True).conjugate().T


def _source_matrix(physics, sources):
    matrix = np.zeros((physics.n, len(sources)), dtype=np.complex128)
    matrix[np.asarray(sources), np.arange(len(sources))] = 1.0 / physics.spacing**2
    return matrix


def _is_transpose_symmetric(physics, model, frequency):
    operator = physics.operator(model, frequency)
    difference = operator - operator.T
    return not difference.nnz or not np.any(difference.data != 0)


def exact_survey_objective_gradient(
    *, physics: Helmholtz2D, m: np.ndarray, blocks: Iterable[SurveyBlock],
    receiver_indices: Iterable[int],
    strategy: Literal["best", "ordinary", "symmetry", "coded"] = "best",
) -> ExactSurveyGradientResult:
    r"""Select or force an exact full-gradient route at each frequency.

    The implemented RHS alternatives are ordinary batched forward/adjoint
    ``2s``, residual-coded receiver ``2r``, and, only with verified transpose
    symmetry, receiver/source-state reuse ``r+s-c``. Here r counts unique
    receiver positions and c counts sources collocated with those positions.
    ``best`` minimizes these counts per frequency, preferring ordinary then
    symmetry on ties. This is an operation-count choice among these routes,
    not a universal optimality or fastest-runtime assertion.

    Every observed receiver slot remains in the objective. Repeated receiver
    residuals are summed, not averaged, when constructing adjoint/coded RHS.
    Point sources use the existing ``1/spacing²`` reference amplitude. The
    coded route needs no transpose symmetry and stores no source wavefields.
    The other two reference routes use batched source fields; no field-scale
    memory or alternate-physics claim is made.
    """
    if strategy not in {"best", "ordinary", "symmetry", "coded"}:
        raise ValueError("strategy must be best, ordinary, symmetry, or coded")
    model, grouped, unique, expand = _inputs(physics, m, blocks, receiver_indices)
    receivers = np.asarray([unique[i] for i in expand])
    positions = {value: i for i, value in enumerate(unique)}
    objective, gradient, expected_rhs = 0.0, np.zeros(physics.n), 0
    strategies = {}
    amplitude = 1.0 / physics.spacing**2
    for frequency, group in grouped.items():
        sources = [int(block.source_index) for block in group]
        observed = np.stack([block.observed for block in group], axis=1)
        s, r = len(sources), len(unique)
        covered = sum(source in positions for source in sources)
        counts = {"ordinary": 2*s, "coded": 2*r}
        symmetric = _is_transpose_symmetric(physics, model, frequency) if strategy in {"best", "symmetry"} else False
        if symmetric:
            # Insertion order implements the stated deterministic tie policy.
            counts = {"ordinary": 2*s, "symmetry": r+s-covered, "coded": 2*r}
        if strategy == "symmetry" and not symmetric:
            raise ValueError("the symmetry route requires A.T == A; the coded route does not")
        chosen = min(counts, key=counts.get) if strategy == "best" else strategy
        strategies[frequency] = chosen
        expected_rhs += counts[chosen]
        omega2 = (2*np.pi*frequency)**2
        if chosen == "ordinary":
            state = physics._solve(model, _source_matrix(physics, sources), frequency, category="forward_rhs")
            residual = state[receivers, :] - observed
            rhs = np.zeros_like(state)
            np.add.at(rhs, receivers, residual)
            adjoint = physics._solve(model, rhs, frequency, category="adjoint_rhs", adjoint=True)
            contribution = omega2*np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
        else:
            c = _receiver_map(physics, model, frequency, unique)
            if chosen == "coded":
                residual = (amplitude*c[:, sources])[expand, :] - observed
                accumulated = np.zeros((r, s), dtype=np.complex128)
                np.add.at(accumulated, expand, residual)
                rhs = np.zeros((physics.n, r), dtype=np.complex128)
                np.add.at(rhs, np.asarray(sources), amplitude*accumulated.conjugate().T)
                coded = physics._solve(model, rhs, frequency, category="forward_rhs")
                contribution = omega2*np.real(np.sum(c.T*coded, axis=1))
            else:
                state = np.empty((physics.n, s), dtype=np.complex128)
                missing = [i for i, source in enumerate(sources) if source not in positions]
                for i, source in enumerate(sources):
                    if source in positions:
                        state[:, i] = amplitude*c[positions[source], :]
                if missing:
                    state[:, missing] = physics._solve(model,
                        _source_matrix(physics, [sources[i] for i in missing]), frequency, category="forward_rhs")
                residual = state[receivers, :] - observed
                accumulated = np.zeros((r, s), dtype=np.complex128)
                np.add.at(accumulated, expand, residual)
                adjoint = c.conjugate().T @ accumulated
                contribution = omega2*np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
        objective += 0.5*float(np.vdot(residual, residual).real)
        gradient += contribution
    objective, gradient = _finite_result(objective, gradient)
    return ExactSurveyGradientResult(objective, gradient, strategies, expected_rhs)


def residual_coded_objective_gradient(
    *, physics: Helmholtz2D, m: np.ndarray, blocks: Iterable[SurveyBlock],
    receiver_indices: Iterable[int],
) -> ExactSurveyGradientResult:
    """Force the exact 2r receiver-coded route without a symmetry assumption."""
    return exact_survey_objective_gradient(physics=physics, m=m, blocks=blocks,
        receiver_indices=receiver_indices, strategy="coded")


def best_exact_objective_gradient(
    *, physics: Helmholtz2D, m: np.ndarray, blocks: Iterable[SurveyBlock],
    receiver_indices: Iterable[int],
) -> ExactSurveyGradientResult:
    """Choose min(2s, 2r, r+s-c when transpose symmetry is valid)."""
    return exact_survey_objective_gradient(physics=physics, m=m, blocks=blocks,
        receiver_indices=receiver_indices, strategy="best")
