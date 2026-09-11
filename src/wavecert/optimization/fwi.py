from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np

from wavecert.experiments.synthetic import velocity_to_m
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.repair.adaptive import adaptive_hybrid_direction, global_fallback_direction
from wavecert.surrogates.base import WavefieldSurrogate

Array = np.ndarray


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    true_objective: float
    model_relative_l2: float
    method_objective: float
    exact_gradient_blocks: int
    cumulative_exact_gradient_blocks: int
    step_size: float
    accepted: bool
    mode: str
    certificate_rounds: int
    algorithm_seconds: float
    validation_seconds: float


@dataclass(frozen=True)
class InversionRun:
    policy: str
    final_model: Array
    records: tuple[IterationRecord, ...]
    total_exact_gradient_blocks: int
    total_algorithm_seconds: float
    total_validation_seconds: float

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "final_model": np.asarray(self.final_model).tolist(),
            "records": [asdict(r) for r in self.records],
            "total_exact_gradient_blocks": self.total_exact_gradient_blocks,
            "total_algorithm_seconds": self.total_algorithm_seconds,
            "total_validation_seconds": self.total_validation_seconds,
        }


def exact_objective_gradient(
    physics: Helmholtz2D,
    m: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
) -> tuple[float, Array]:
    objective = 0.0
    gradient = np.zeros(physics.n, dtype=float)
    for block in blocks:
        q = physics.source(block.source_index)
        value, grad, _, _ = physics.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        objective += value
        gradient += grad
    return float(objective), gradient


def exact_objective(
    physics: Helmholtz2D,
    m: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
) -> float:
    total = 0.0
    for block in blocks:
        q = physics.source(block.source_index)
        u = physics.solve_state(m, q, block.frequency_hz)
        residual = physics.restrict(u, receiver_indices) - np.asarray(block.observed)
        total += 0.5 * float(np.vdot(residual, residual).real)
    return float(total)


def surrogate_objective_gradient(
    surrogate: WavefieldSurrogate,
    physics: Helmholtz2D,
    m: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
) -> tuple[float, Array]:
    objective = 0.0
    gradient = np.zeros(physics.n, dtype=float)
    for block in blocks:
        q = physics.source(block.source_index)
        value, grad, _, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        objective += value
        gradient += grad
    return float(objective), gradient


def surrogate_objective(
    surrogate: WavefieldSurrogate,
    physics: Helmholtz2D,
    m: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
) -> float:
    total = 0.0
    for block in blocks:
        q = physics.source(block.source_index)
        value, _, _, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        total += value
    return float(total)


def _normalised_negative(gradient: Array) -> Array:
    gradient = np.asarray(gradient, dtype=float)
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-14:
        return np.zeros_like(gradient)
    return -gradient / norm


def _project_squared_slowness(m: Array, *, vmin: float, vmax: float) -> Array:
    lower = 1.0 / float(vmax) ** 2
    upper = 1.0 / float(vmin) ** 2
    return np.clip(np.asarray(m, dtype=float), lower, upper)


def _model_relative_l2(m: Array, m_true: Array) -> float:
    a = np.asarray(m, dtype=float)
    b = np.asarray(m_true, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def _backtrack(
    m: Array,
    direction: Array,
    objective_fn: Callable[[Array], float],
    current_value: float,
    *,
    initial_step: float,
    vmin: float,
    vmax: float,
    max_trials: int = 8,
) -> tuple[Array, float, float, bool]:
    """Backtrack using the policy's own merit objective.

    True-PDE objective values used for manuscript evaluation are deliberately
    *not* consulted here for neural/WaveCert policies; otherwise the optimizer
    would silently pay full exact-forward cost at every trial.
    """

    step = float(initial_step)
    for _ in range(max_trials):
        candidate = _project_squared_slowness(m + step * direction, vmin=vmin, vmax=vmax)
        value = float(objective_fn(candidate))
        if np.isfinite(value) and value < current_value:
            return candidate, step, value, True
        step *= 0.5
    candidate = _project_squared_slowness(m + step * direction, vmin=vmin, vmax=vmax)
    value = float(objective_fn(candidate))
    return candidate, step, value, bool(np.isfinite(value) and value < current_value)


def run_inversion(
    *,
    policy: str,
    physics: Helmholtz2D,
    surrogate: WavefieldSurrogate,
    m0: Array,
    m_true: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
    iterations: int = 10,
    initial_step: float = 0.02,
    vmin: float = 1.55,
    vmax: float = 3.20,
    case_scale: float = 1.0,
    block_scales: dict[str, float] | None = None,
) -> InversionRun:
    """Run one complete inversion trajectory for a named policy.

    Policies are ``exact``, ``neural-only``, ``conformal-global`` and
    ``conformal-selective``. Exact objective/gradient evaluations performed only
    for *offline validation* are timed separately and do not contribute to the
    policy's exact-gradient-block budget.
    """

    blocks = tuple(blocks)
    m = physics.validate_model(m0).copy()
    m_true = physics.validate_model(m_true)
    cumulative_exact = 0
    records: list[IterationRecord] = []
    algorithm_seconds_total = 0.0
    validation_seconds_total = 0.0

    def exact_merit(candidate: Array) -> float:
        return exact_objective(physics, candidate, blocks, receiver_indices)

    def surrogate_merit(candidate: Array) -> float:
        return surrogate_objective(surrogate, physics, candidate, blocks, receiver_indices)

    for iteration in range(iterations + 1):
        t_val = time.perf_counter()
        true_value = exact_objective(physics, m, blocks, receiver_indices)
        validation_seconds = time.perf_counter() - t_val
        validation_seconds_total += validation_seconds

        if iteration == iterations:
            records.append(
                IterationRecord(
                    iteration=iteration,
                    true_objective=true_value,
                    model_relative_l2=_model_relative_l2(m, m_true),
                    method_objective=float("nan"),
                    exact_gradient_blocks=0,
                    cumulative_exact_gradient_blocks=cumulative_exact,
                    step_size=0.0,
                    accepted=True,
                    mode="final-evaluation",
                    certificate_rounds=0,
                    algorithm_seconds=0.0,
                    validation_seconds=validation_seconds,
                )
            )
            break

        t_alg = time.perf_counter()
        exact_used = 0
        rounds = 0
        mode = policy

        if policy == "exact":
            method_value, grad = exact_objective_gradient(physics, m, blocks, receiver_indices)
            direction = _normalised_negative(grad)
            exact_used = len(blocks)
            merit = exact_merit
        elif policy == "neural-only":
            method_value, grad = surrogate_objective_gradient(
                surrogate, physics, m, blocks, receiver_indices
            )
            direction = _normalised_negative(grad)
            merit = surrogate_merit
        elif policy == "conformal-global":
            method_value, _ = surrogate_objective_gradient(
                surrogate, physics, m, blocks, receiver_indices
            )
            result = global_fallback_direction(
                physics=physics,
                surrogate=surrogate,
                m=m,
                blocks=blocks,
                receiver_indices=receiver_indices,
                stability_mode="directional",
                bound_scale=case_scale,
            )
            direction = result.direction
            exact_used = result.exact_block_evaluations
            rounds = result.certificate_rounds
            mode = result.mode
            merit = surrogate_merit
        elif policy == "conformal-selective":
            method_value, _ = surrogate_objective_gradient(
                surrogate, physics, m, blocks, receiver_indices
            )
            result = adaptive_hybrid_direction(
                physics=physics,
                surrogate=surrogate,
                m=m,
                blocks=blocks,
                receiver_indices=receiver_indices,
                stability_mode="directional",
                bound_scales=block_scales,
            )
            direction = result.direction
            exact_used = result.exact_block_evaluations
            rounds = result.certificate_rounds
            mode = result.mode
            merit = surrogate_merit
        else:
            raise ValueError(f"unknown policy: {policy}")

        if np.linalg.norm(direction) <= 1e-14:
            candidate = m.copy()
            step = 0.0
            accepted = False
        else:
            candidate, step, _, accepted = _backtrack(
                m,
                direction,
                merit,
                method_value,
                initial_step=initial_step,
                vmin=vmin,
                vmax=vmax,
            )
            if not accepted:
                # A failed merit backtrack should not force a knowingly bad
                # finite step. Keep the current iterate and let the trajectory
                # record the stall explicitly.
                candidate = m.copy()
                step = 0.0

        algorithm_seconds = time.perf_counter() - t_alg
        algorithm_seconds_total += algorithm_seconds
        cumulative_exact += exact_used
        records.append(
            IterationRecord(
                iteration=iteration,
                true_objective=true_value,
                model_relative_l2=_model_relative_l2(m, m_true),
                method_objective=method_value,
                exact_gradient_blocks=exact_used,
                cumulative_exact_gradient_blocks=cumulative_exact,
                step_size=step,
                accepted=accepted,
                mode=mode,
                certificate_rounds=rounds,
                algorithm_seconds=algorithm_seconds,
                validation_seconds=validation_seconds,
            )
        )
        m = candidate

    return InversionRun(
        policy=policy,
        final_model=m,
        records=tuple(records),
        total_exact_gradient_blocks=cumulative_exact,
        total_algorithm_seconds=algorithm_seconds_total,
        total_validation_seconds=validation_seconds_total,
    )


def velocity_from_m(m: Array) -> Array:
    return 1.0 / np.sqrt(np.asarray(m, dtype=float))


def m_from_velocity(velocity: Array) -> Array:
    return velocity_to_m(np.asarray(velocity, dtype=float)).reshape(-1)
