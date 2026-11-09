from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace

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
    step_acceptance: str = "surrogate"
    # Requested acceptance stays above; exact recovery can override legacy
    # surrogate merit. Record its actual paid criterion separately.
    effective_step_acceptance: str | None = None
    direction_certified: bool = False
    guarantee_scope: str = "unverified-direction"
    line_search_trials: int = 0
    verification_gradient_blocks: int = 0
    online_exact_cost: dict[str, int] = field(default_factory=dict)
    validation_exact_cost: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class InversionRun:
    policy: str
    final_model: Array
    records: tuple[IterationRecord, ...]
    total_exact_gradient_blocks: int
    total_algorithm_seconds: float
    total_validation_seconds: float
    step_acceptance: str = "surrogate"
    total_verification_gradient_blocks: int = 0
    total_online_exact_cost: dict[str, int] = field(default_factory=dict)
    total_validation_exact_cost: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["final_model"] = np.asarray(self.final_model).tolist()
        return result


def _cost_snapshot(physics: Helmholtz2D) -> dict[str, int]:
    method = getattr(physics, "cost_snapshot", None)
    return {} if method is None else method()


def _cost_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in sorted(set(before) | set(after))}


def _sum_costs(records: Iterable[IterationRecord], field_name: str) -> dict[str, int]:
    total: dict[str, int] = {}
    for record in records:
        for key, value in getattr(record, field_name).items():
            total[key] = total.get(key, 0) + int(value)
    return total


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
    """Evaluate the forward-only surrogate merit; no gradient is needed."""
    total = 0.0
    for block in blocks:
        objective = getattr(surrogate, "objective", None)
        if callable(objective):
            total += float(objective(
                m, physics.source(block.source_index), block.observed,
                receiver_indices, block.frequency_hz,
            ))
            continue
        state = surrogate.state(m, physics.source(block.source_index), block.frequency_hz)
        residual = physics.restrict(state, receiver_indices) - np.asarray(block.observed)
        total += 0.5 * float(np.vdot(residual, residual).real)
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
    gradient: Array | None = None,
    armijo_constant: float = 1e-4,
    trial_counter: list[int] | None = None,
) -> tuple[Array, float, float, bool]:
    """Accept a projected decrease, optionally using a true Armijo test.

    When ``gradient`` is supplied it must be the gradient of ``objective_fn``
    at ``m``. Armijo uses ``gradient @ (candidate-m)``, not the unprojected
    direction: clipping can change the direction or give zero displacement.
    A failed search returns the original model and a zero step.
    """
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step must be finite and positive")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie strictly between zero and one")
    if gradient is not None:
        gradient = np.asarray(gradient, dtype=float)
    step = float(initial_step)
    for _ in range(max_trials):
        candidate = _project_squared_slowness(m + step * direction, vmin=vmin, vmax=vmax)
        displacement = candidate - m
        if not np.any(displacement):
            step *= 0.5
            continue
        slope = None if gradient is None else float(np.dot(gradient, displacement))
        if slope is not None and (not np.isfinite(slope) or slope >= 0.0):
            step *= 0.5
            continue
        if trial_counter is not None:
            trial_counter[0] += 1
        value = float(objective_fn(candidate))
        threshold = current_value if slope is None else current_value + armijo_constant * slope
        if np.isfinite(value) and value < current_value and value <= threshold:
            return candidate, step, value, True
        step *= 0.5
    return m.copy(), 0.0, float(current_value), False


def _heuristic_direction(
    *,
    policy: str,
    physics: Helmholtz2D,
    surrogate: WavefieldSurrogate,
    m: Array,
    blocks: tuple[SurveyBlock, ...],
    receiver_indices: tuple[int, ...],
    repair_blocks: int,
    rng: np.random.Generator,
) -> tuple[Array, float, int]:
    """Fixed-budget non-oracle repair. Residual selection uses no exact state."""
    gradients: list[Array] = []
    scores: list[float] = []
    value = 0.0
    for block in blocks:
        q = physics.source(block.source_index)
        block_value, grad, state, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        value += float(block_value)
        gradients.append(np.asarray(grad, dtype=float))
        if policy == "residual-repair":
            residual = physics.operator(m, block.frequency_hz) @ state - q
            scores.append(float(np.linalg.norm(residual) / max(np.linalg.norm(q), 1e-15)))
    if policy == "random-repair":
        chosen = rng.choice(len(blocks), size=repair_blocks, replace=False)
    else:
        chosen = np.argsort(-np.asarray(scores), kind="stable")[:repair_blocks]
    for index in chosen:
        block = blocks[int(index)]
        _, grad, _, _ = physics.objective_and_gradient(
            m, physics.source(block.source_index), block.observed, receiver_indices, block.frequency_hz
        )
        gradients[int(index)] = np.asarray(grad, dtype=float)
    return _normalised_negative(np.sum(gradients, axis=0)), value, int(len(chosen))


def _calibration_domain_matches(
    domain: dict[str, object] | None, physics: Helmholtz2D,
    blocks: tuple[SurveyBlock, ...], receiver_indices: tuple[int, ...],
) -> bool:
    """Validate an acquisition-support contract and check observable support.

    This checks only frequency/source membership and the exact ordered receiver
    tuple. It cannot detect geological shift, model-state shift, or dependence
    introduced by adaptive inversion trajectories.
    """
    fields = {"frequencies_hz", "source_indices", "receiver_indices"}
    if not isinstance(domain, dict) or set(domain) != fields:
        raise ValueError("guarded calibration_domain must declare exactly frequencies_hz, source_indices and receiver_indices")
    try:
        frequencies = np.asarray(domain["frequencies_hz"], dtype=float)
        sources = np.asarray(domain["source_indices"])
        receivers = np.asarray(domain["receiver_indices"])
        if frequencies.ndim != 1 or frequencies.size == 0 or np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("calibration frequencies must be nonempty, finite and positive")
        for label, indices in (("source", sources), ("receiver", receivers)):
            if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
                    or np.any(indices < 0) or np.any(indices >= physics.n)):
                raise ValueError(f"calibration {label} indices must be nonempty integer indices within the physical grid")
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid calibration domain arrays") from exc
    return (
        all(float(block.frequency_hz) in set(frequencies) and int(block.source_index) in set(sources) for block in blocks)
        and tuple(int(index) for index in receivers) == tuple(receiver_indices)
    )


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
    step_acceptance: str = "surrogate",
    random_seed: int = 0,
    repair_blocks: int = 1,
    periodic_interval: int = 4,
    repair_schedule: tuple[int, ...] | None = None,
    max_line_search_trials: int = 12,
    armijo_constant: float = 1e-4,
    calibration_domain: dict[str, object] | None = None,
) -> InversionRun:
    """Run an inversion with explicit direction and finite-step semantics.

    ``surrogate`` preserves legacy merit-based acceptance: exact FWI uses its
    exact merit, all other policies use surrogate merit, with no true-step
    guarantee. ``exact-decrease`` verifies strict decrease of the true discrete
    objective using forward solves. ``exact-armijo`` additionally obtains the
    true gradient and tests sufficient decrease along the *projected*
    displacement. Its full verification gradient cost is online and counted.

    Exact-policy, acquisition-mismatch, unavailable-analytical-stability and
    zero/invalid-hybrid-gradient recovery use exact merit even when legacy
    ``surrogate`` acceptance was requested. In particular a dead neural merit
    cannot block recovery by a valid exact gradient. Records retain the
    requested ``step_acceptance`` and state the actual paid criterion in
    ``effective_step_acceptance``. Ordinary legacy repair keeps its original
    surrogate merit.

    Deterministic and conformal direction certificates are distinct; neither
    certifies an arbitrary finite projected step. Uncertified adaptive outputs
    are rejected. ``repair_schedule`` is an externally specified per-iteration
    budget; deriving it from test-set WaveCert outcomes is a retrospective
    oracle diagnostic, not a deployable matched-cost baseline.

    ``conformal-selective-guarded`` additionally requires an explicit
    acquisition-domain contract. Frequency/source support or ordered receiver
    mismatch abstains before any surrogate/certificate work and uses exact
    gradients with exact merit. This only detects observable acquisition
    mismatch; it establishes no geology-shift or trajectory-wide guarantee.

    Offline validation uses a separate solver instance so its factorizations
    cannot warm the online policy's cache. Run costs include inference,
    certificates and online step checks; counters distinguish all exact solves
    from the narrower repaired-gradient-block count.
    """
    allowed = {
        "exact", "neural-only", "conformal-global", "conformal-selective",
        "deterministic-global", "deterministic-selective", "random-repair",
        "residual-repair", "periodic-exact", "analytic-selective", "conformal-selective-guarded",
    }
    if policy not in allowed:
        raise ValueError(f"unknown policy: {policy}")
    if step_acceptance not in {"surrogate", "exact-decrease", "exact-armijo"}:
        raise ValueError("step_acceptance must be surrogate, exact-decrease, or exact-armijo")
    if iterations < 0 or int(iterations) != iterations:
        raise ValueError("iterations must be a non-negative integer")
    if not (np.isfinite(vmin) and np.isfinite(vmax) and 0 < vmin < vmax):
        raise ValueError("velocity bounds must be finite and satisfy 0 < vmin < vmax")
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step must be finite and positive")
    if periodic_interval < 1 or int(periodic_interval) != periodic_interval:
        raise ValueError("periodic_interval must be a positive integer")
    if max_line_search_trials < 1 or int(max_line_search_trials) != max_line_search_trials:
        raise ValueError("max_line_search_trials must be a positive integer")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie strictly between zero and one")
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one survey block is required")
    domain_mismatch = policy == "conformal-selective-guarded" and not _calibration_domain_matches(
        calibration_domain, physics, blocks, receiver_indices,
    )
    if int(repair_blocks) != repair_blocks or not 0 <= repair_blocks <= len(blocks):
        raise ValueError("repair_blocks must be an integer between zero and the number of blocks")
    if repair_schedule is not None:
        if len(repair_schedule) != iterations:
            raise ValueError("repair_schedule must contain exactly one budget per iteration")
        if any(int(k) != k or not 0 <= k <= len(blocks) for k in repair_schedule):
            raise ValueError("repair_schedule budgets must be integers within the block count")
    m = physics.validate_model(m0).copy()
    if not np.allclose(_project_squared_slowness(m, vmin=vmin, vmax=vmax), m, rtol=0, atol=1e-14):
        raise ValueError("initial model must lie within the velocity bounds")
    m_true = physics.validate_model(m_true)
    validation_physics = replace(physics)
    clear_cache = getattr(physics, "clear_solver_cache", None)
    if clear_cache is not None:
        clear_cache()
    cumulative_exact = 0
    records: list[IterationRecord] = []
    rng = np.random.default_rng(random_seed)

    def exact_merit(candidate: Array) -> float:
        return exact_objective(physics, candidate, blocks, receiver_indices)

    def surrogate_merit(candidate: Array) -> float:
        return surrogate_objective(surrogate, physics, candidate, blocks, receiver_indices)

    for iteration in range(iterations + 1):
        val_before = _cost_snapshot(validation_physics)
        t_val = time.perf_counter()
        true_value = exact_objective(validation_physics, m, blocks, receiver_indices)
        validation_seconds = time.perf_counter() - t_val
        validation_cost = _cost_delta(val_before, _cost_snapshot(validation_physics))
        if iteration == iterations:
            records.append(IterationRecord(
                iteration=iteration, true_objective=true_value,
                model_relative_l2=_model_relative_l2(m, m_true), method_objective=float("nan"),
                exact_gradient_blocks=0, cumulative_exact_gradient_blocks=cumulative_exact,
                step_size=0.0, accepted=False, mode="final-evaluation", certificate_rounds=0,
                algorithm_seconds=0.0, validation_seconds=validation_seconds,
                step_acceptance=step_acceptance, validation_exact_cost=validation_cost,
                effective_step_acceptance=step_acceptance,
            ))
            break

        online_before = _cost_snapshot(physics)
        t_alg = time.perf_counter()
        exact_used = 0
        verification_used = 0
        rounds = 0
        mode = policy
        direction_certified = False
        guarantee_scope = "unverified-direction"
        effective_acceptance = step_acceptance
        true_gradient: Array | None = None
        online_true_value: float | None = None
        adaptive_ok = True
        if policy == "exact" or domain_mismatch or (policy == "periodic-exact" and iteration % periodic_interval == 0):
            method_value, grad = exact_objective_gradient(physics, m, blocks, receiver_indices)
            direction = _normalised_negative(grad)
            exact_used = len(blocks)
            true_gradient, online_true_value = grad, method_value
            direction_certified = bool(np.linalg.norm(grad) > 1e-14)
            guarantee_scope = "exact-gradient-direction"
            merit = exact_merit if policy == "exact" or domain_mismatch else surrogate_merit
            if step_acceptance == "surrogate" and (policy == "exact" or domain_mismatch):
                effective_acceptance = "exact-decrease"
            if domain_mismatch:
                mode = "domain-mismatch-exact-fallback"
            if policy == "periodic-exact":
                mode = "scheduled-exact"
                method_value = surrogate_merit(m)
        elif policy in {"neural-only", "periodic-exact"}:
            method_value, grad = surrogate_objective_gradient(surrogate, physics, m, blocks, receiver_indices)
            direction = _normalised_negative(grad)
            merit = surrogate_merit
            if policy == "periodic-exact":
                mode = "scheduled-neural"
        elif policy in {"random-repair", "residual-repair"}:
            budget = repair_blocks if repair_schedule is None else repair_schedule[iteration]
            direction, method_value, exact_used = _heuristic_direction(
                policy=policy, physics=physics, surrogate=surrogate, m=m, blocks=blocks,
                receiver_indices=receiver_indices, repair_blocks=int(budget), rng=rng,
            )
            merit = surrogate_merit
        else:
            # Defer merit work until the certificate result: an analytic
            # stability preflight may abstain without evaluating the surrogate.
            method_value = float("nan")
            deterministic = policy.startswith("deterministic-") or policy == "analytic-selective"
            stability_mode = "analytic" if policy == "analytic-selective" else "directional"
            if policy.endswith("global"):
                result = global_fallback_direction(
                    physics=physics, surrogate=surrogate, m=m, blocks=blocks,
                    receiver_indices=receiver_indices, stability_mode=stability_mode,
                    bound_scale=1.0 if deterministic else case_scale,
                )
            else:
                result = adaptive_hybrid_direction(
                    physics=physics, surrogate=surrogate, m=m, blocks=blocks,
                    receiver_indices=receiver_indices, stability_mode=stability_mode,
                    bound_scales=None if deterministic else block_scales,
                )
            direction = result.direction
            exact_used = result.exact_block_evaluations
            rounds = result.certificate_rounds
            mode = result.mode
            adaptive_ok = result.certified_descent
            direction_certified = result.certified_descent
            if exact_used == len(blocks) and adaptive_ok:
                # Full fallback already paid for every gradient block. Reuse
                # its exact total rather than charge a second verification.
                true_gradient = -result.final_hybrid_gradient_norm * direction
                paid_objective = getattr(result, "exact_objective", None)
                if paid_objective is not None:
                    if not np.isfinite(paid_objective):
                        raise ValueError("full exact fallback returned a nonfinite objective")
                    online_true_value = float(paid_objective)
            exact_recovery = result.mode in {
                "zero-hybrid-gradient-exact-fallback", "invalid-hybrid-gradient-exact-fallback",
            }
            guarantee_scope = (
                "exact-gradient-direction" if exact_recovery else
                "discrete-deterministic-direction" if deterministic else "calibrated-direction"
            )
            if not adaptive_ok:
                mode = f"rejected-{mode}"
            if exact_recovery or not getattr(result, "surrogate_gradient_evaluated", True):
                merit = exact_merit
                if step_acceptance == "surrogate":
                    effective_acceptance = "exact-decrease"
                if adaptive_ok:
                    if online_true_value is None:
                        online_true_value = exact_merit(m)
                    method_value = online_true_value
            else:
                merit = surrogate_merit
                if step_acceptance == "surrogate":
                    method_value = surrogate_merit(m)

        armijo_gradient = None
        if step_acceptance == "exact-armijo" and adaptive_ok:
            if true_gradient is None:
                online_true_value, true_gradient = exact_objective_gradient(physics, m, blocks, receiver_indices)
                verification_used = len(blocks)
            elif online_true_value is None:
                online_true_value = exact_merit(m)
            merit, method_value, armijo_gradient = exact_merit, float(online_true_value), true_gradient
        elif step_acceptance == "exact-decrease" and adaptive_ok:
            # Never borrow the offline validation value for an online decision.
            if online_true_value is None:
                online_true_value = exact_merit(m)
            merit, method_value = exact_merit, float(online_true_value)

        trials = [0]
        if not adaptive_ok or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1e-14:
            candidate, step, accepted = m.copy(), 0.0, False
        else:
            candidate, step, _, accepted = _backtrack(
                m, direction, merit, method_value, initial_step=initial_step,
                vmin=vmin, vmax=vmax, max_trials=max_line_search_trials,
                gradient=armijo_gradient, armijo_constant=armijo_constant, trial_counter=trials,
            )
        algorithm_seconds = time.perf_counter() - t_alg
        online_cost = _cost_delta(online_before, _cost_snapshot(physics))
        cumulative_exact += exact_used
        records.append(IterationRecord(
            iteration=iteration, true_objective=true_value,
            model_relative_l2=_model_relative_l2(m, m_true), method_objective=float(method_value),
            exact_gradient_blocks=exact_used, cumulative_exact_gradient_blocks=cumulative_exact,
            step_size=step, accepted=accepted, mode=mode, certificate_rounds=rounds,
            algorithm_seconds=algorithm_seconds, validation_seconds=validation_seconds,
            step_acceptance=step_acceptance, direction_certified=direction_certified,
            effective_step_acceptance=effective_acceptance,
            guarantee_scope=guarantee_scope, line_search_trials=trials[0],
            verification_gradient_blocks=verification_used, online_exact_cost=online_cost,
            validation_exact_cost=validation_cost,
        ))
        m = candidate

    return InversionRun(
        policy=policy, final_model=m, records=tuple(records),
        total_exact_gradient_blocks=cumulative_exact,
        total_algorithm_seconds=float(sum(r.algorithm_seconds for r in records)),
        total_validation_seconds=float(sum(r.validation_seconds for r in records)),
        step_acceptance=step_acceptance,
        total_verification_gradient_blocks=sum(r.verification_gradient_blocks for r in records),
        total_online_exact_cost=_sum_costs(records, "online_exact_cost"),
        total_validation_exact_cost=_sum_costs(records, "validation_exact_cost"),
    )


def velocity_from_m(m: Array) -> Array:
    return 1.0 / np.sqrt(np.asarray(m, dtype=float))


def m_from_velocity(velocity: Array) -> Array:
    return velocity_to_m(np.asarray(velocity, dtype=float)).reshape(-1)
