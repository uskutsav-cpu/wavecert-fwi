"""Exact receiver-map controls for the small frequency-domain reference solver.

These are established linear-algebra/reciprocity controls, not neural methods
or novel WaveCert certificates. They use ordinary floating-point solves, not
verified roundoff enclosures. A negative directional derivative does not by
itself guarantee decrease after a finite or projected update.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock


@dataclass(frozen=True)
class ReciprocityGradientResult:
    objective: float
    gradient: np.ndarray
    frequency_strategies: dict[float, str]
    expected_rhs: int


def _real_vector(value: np.ndarray, n: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (n,) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real vector of shape ({n},)")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _index(value: int, n: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not 0 <= value < n):
        raise ValueError(f"{name} must be an integer in [0, {n})")
    return int(value)


def _inputs(
    physics: Helmholtz2D, m: np.ndarray, blocks: Iterable[SurveyBlock],
    receiver_indices: Iterable[int],
) -> tuple[np.ndarray, dict[float, tuple[SurveyBlock, ...]], tuple[int, ...], np.ndarray]:
    if not isinstance(physics, Helmholtz2D):
        raise TypeError("these controls require the Helmholtz2D reference backend")
    model = physics.validate_model(_real_vector(m, physics.n, "model"))
    receivers = tuple(_index(i, physics.n, "receiver index") for i in receiver_indices)
    if not receivers:
        raise ValueError("at least one receiver is required")
    unique = tuple(dict.fromkeys(receivers))
    positions = {value: i for i, value in enumerate(unique)}
    expand = np.asarray([positions[i] for i in receivers], dtype=int)
    grouped: dict[float, list[SurveyBlock]] = {}
    seen: set[tuple[float, int]] = set()
    for block in blocks:
        source = _index(block.source_index, physics.n, "source index")
        frequency = float(block.frequency_hz)
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequencies must be finite and positive")
        observed = np.asarray(block.observed)
        if observed.shape != (len(receivers),) or not np.all(np.isfinite(observed)):
            raise ValueError("observed data must be a finite vector matching the receivers")
        key = (frequency, source)
        if key in seen:
            raise ValueError("source/frequency blocks must be unique")
        seen.add(key)
        grouped.setdefault(frequency, []).append(block)
    if not grouped:
        raise ValueError("at least one source/frequency block is required")
    return model, {f: tuple(grouped[f]) for f in sorted(grouped)}, unique, expand


def receiver_map_directional_oracle(
    *, physics: Helmholtz2D, m: np.ndarray, blocks: Iterable[SurveyBlock],
    receiver_indices: Iterable[int], direction: np.ndarray,
) -> tuple[float, float]:
    r"""Evaluate exact discrete objective and derivative with two receiver maps.

    With ``C=P A^-1`` and ``T=C diag(v) A^-1``, the exact receiver state and
    tangent are ``C q`` and ``omega² T q``. Each unique receiver costs two
    RHS per frequency, independent of source count. No neural approximation,
    SVD, or stability bound is used. Repeated samples remain repeated in the
    objective; redundant map solves are removed. This identity does not rely
    on Hermitian or transpose symmetry, but its implementation here is scoped
    to the reference backend rather than Devito or another wave operator.
    """
    model, grouped, unique, expand = _inputs(physics, m, blocks, receiver_indices)
    vector = _real_vector(direction, physics.n, "direction")
    objective = derivative = 0.0
    amplitude = 1.0 / physics.spacing**2
    for frequency, group in grouped.items():
        state_map = physics.receiver_resolvent_matrix(model, frequency, unique)
        tangent_map = physics.directional_receiver_tangent_resolvent_matrix(
            model, frequency, unique, vector, receiver_resolvent_matrix=state_map,
        )
        omega2 = (2 * np.pi * frequency)**2
        for block in group:
            predicted = amplitude * state_map[expand, block.source_index]
            tangent = omega2 * amplitude * tangent_map[expand, block.source_index]
            residual = predicted - np.asarray(block.observed)
            objective += 0.5 * float(np.vdot(residual, residual).real)
            derivative += float(np.vdot(residual, tangent).real)
    if not np.isfinite(objective) or not np.isfinite(derivative):
        raise RuntimeError("receiver-map objective or derivative is non-finite")
    return float(objective), float(derivative)


def reciprocity_objective_gradient(
    *, physics: Helmholtz2D, m: np.ndarray, blocks: Iterable[SurveyBlock],
    receiver_indices: Iterable[int],
) -> ReciprocityGradientResult:
    r"""Use the cheaper exact full-gradient route at each frequency.

    Ordinary adjoint-state work costs ``2s`` RHS for ``s`` distinct sources.
    Receiver reciprocity costs ``r+s-c`` for ``r`` unique receiver locations
    and ``c`` sources covered by those locations. The latter constructs
    ``C=P A^-1``, forms all adjoints as ``C^H residual``, and obtains covered
    source states from rows of C because ``A.T == A``. Others use a forward
    solve. The ordinary route is retained on ties or when it is cheaper.

    This symmetry is transpose symmetry, NOT Hermitian symmetry: damping
    makes this operator complex and generally non-Hermitian. The operator is
    checked before any reciprocity reuse. Counts describe RHS, not runtime;
    holding receiver maps also incurs memory and matrix multiplication work.
    """
    model, grouped, unique, expand = _inputs(physics, m, blocks, receiver_indices)
    receivers = tuple(unique[i] for i in expand)
    positions = {value: i for i, value in enumerate(unique)}
    objective = 0.0
    gradient = np.zeros(physics.n, dtype=float)
    strategies = {}
    expected_rhs = 0
    amplitude = 1.0 / physics.spacing**2
    for frequency, group in grouped.items():
        s, r = len(group), len(unique)
        covered = sum(block.source_index in positions for block in group)
        reciprocal_rhs = r + s - covered
        if reciprocal_rhs >= 2 * s:
            strategies[frequency] = "ordinary-adjoint"
            expected_rhs += 2 * s
            for block in group:
                value, grad, _, _ = physics.objective_and_gradient(
                    model, physics.source(block.source_index), block.observed,
                    receivers, frequency,
                )
                objective += value
                gradient += grad
            continue

        operator = physics.operator(model, frequency)
        difference = operator - operator.T
        if difference.nnz and np.any(difference.data != 0):
            raise ValueError("receiver reciprocity requires exact transpose symmetry A.T == A")
        strategies[frequency] = "receiver-reciprocity"
        expected_rhs += reciprocal_rhs
        state_map = physics.receiver_resolvent_matrix(model, frequency, unique)
        omega2 = (2 * np.pi * frequency)**2
        for block in group:
            if block.source_index in positions:
                state = amplitude * state_map[positions[block.source_index], :]
            else:
                state = physics.solve_state(model, physics.source(block.source_index), frequency)
            residual = physics.restrict(state, receivers) - np.asarray(block.observed)
            # P may repeat locations: its adjoint must add those residuals.
            accumulated = np.zeros(r, dtype=np.complex128)
            np.add.at(accumulated, expand, residual)
            adjoint = state_map.conjugate().T @ accumulated
            objective += 0.5 * float(np.vdot(residual, residual).real)
            gradient += omega2 * np.real(np.conjugate(adjoint) * state)
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("reciprocity objective or gradient is non-finite")
    return ReciprocityGradientResult(float(objective), gradient, strategies, expected_rhs)
