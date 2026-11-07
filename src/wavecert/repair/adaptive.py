from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from wavecert.certificates.directional import StabilityMode, certify_direction
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.surrogates.base import WavefieldSurrogate

Array = np.ndarray


@dataclass(frozen=True)
class AdaptiveDirectionResult:
    mode: str
    certified_descent: bool
    direction: Array
    repaired_blocks: tuple[str, ...]
    exact_block_evaluations: int
    total_blocks: int
    initial_upper_derivative: float
    final_upper_derivative: float
    surrogate_gradient_norm: float
    final_hybrid_gradient_norm: float
    certificate_rounds: int
    surrogate_gradient_evaluated: bool = True
    # False when inference was skipped or produced an unusable gradient. Its
    # reported norm then uses a finite zero sentinel, not a stationarity claim.
    surrogate_gradient_valid: bool = True
    # Available only after all blocks were evaluated online by exact physics.
    # Reusing it avoids a duplicate current-objective forward pass.
    exact_objective: float | None = None

    @property
    def exact_block_fraction(self) -> float:
        return self.exact_block_evaluations / max(self.total_blocks, 1)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["direction"] = np.asarray(self.direction).tolist()
        data["exact_block_fraction"] = self.exact_block_fraction
        return data


@dataclass
class _BlockState:
    block: SurveyBlock
    q: Array
    neural_gradient: Array
    exact_gradient: Array | None = None
    exact_objective: float | None = None


def _normalised_negative(gradient: Array) -> Array:
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-14:
        return np.zeros_like(np.asarray(gradient, dtype=float))
    return -np.asarray(gradient, dtype=float) / norm


def _validate_scale(value: float) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bound scales must be finite and strictly positive")
    return scale


def _frequency_constants(
    physics: Helmholtz2D,
    m: Array,
    receiver_indices: tuple[int, ...],
    frequencies: Iterable[float],
    *,
    need_beta: bool = False,
    need_receiver: bool = True,
    analytic_beta: bool = False,
) -> dict[float, dict[str, object]]:
    constants: dict[float, dict[str, object]] = {}
    for frequency in sorted({float(v) for v in frequencies}):
        entry: dict[str, object] = {}
        if need_receiver:
            receiver_matrix = physics.receiver_resolvent_matrix(m, frequency, receiver_indices)
            entry.update(
                receiver_matrix=receiver_matrix,
                alpha=physics.receiver_resolvent_norm(
                    m, frequency, receiver_indices, receiver_resolvent_matrix=receiver_matrix,
                ),
            )
        if need_beta:
            entry["beta"] = physics.smallest_singular_value(m, frequency)
        if analytic_beta:
            entry["beta"] = physics.analytical_stability_bound(m, frequency).beta_lower_bound
        constants[frequency] = entry
    return constants


def _exact_fallback(
    *, physics: Helmholtz2D, surrogate_gradient: Array | None, m: Array,
    blocks: tuple[SurveyBlock, ...], receiver_indices: tuple[int, ...], mode: str,
    initial_upper: float = float("inf"), certificate_rounds: int = 0,
    states: list[_BlockState] | None = None,
) -> AdaptiveDirectionResult:
    gradients = []
    objective = 0.0
    for index, block in enumerate(blocks):
        state = None if states is None else states[index]
        if state is not None and state.exact_gradient is not None:
            value, gradient = state.exact_objective, state.exact_gradient
        else:
            value, gradient, _, _ = physics.objective_and_gradient(
                m, physics.source(block.source_index), block.observed, receiver_indices,
                block.frequency_hz,
            )
        if value is None or not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ValueError("exact fallback returned a nonfinite objective or gradient")
        objective += float(value)
        gradients.append(gradient)
    total = np.sum(gradients, axis=0)
    norm = float(np.linalg.norm(total))
    if not np.isfinite(norm):
        raise ValueError("exact fallback gradient norm must be finite")
    direction = _normalised_negative(total)
    has_direction = norm > 1e-14
    surrogate_norm = (
        None if surrogate_gradient is None else float(np.linalg.norm(surrogate_gradient))
    )
    surrogate_valid = surrogate_norm is not None and np.isfinite(surrogate_norm)
    return AdaptiveDirectionResult(
        mode=mode, certified_descent=has_direction, direction=direction,
        repaired_blocks=tuple(block.label for block in blocks),
        exact_block_evaluations=len(blocks), total_blocks=len(blocks),
        initial_upper_derivative=initial_upper,
        final_upper_derivative=-norm if has_direction else 0.0,
        # A preflight can skip the neural gradient; failed inference can make
        # it invalid. Separate flags disambiguate their finite zero sentinel.
        surrogate_gradient_norm=(
            surrogate_norm if surrogate_valid else 0.0
        ),
        final_hybrid_gradient_norm=norm, certificate_rounds=certificate_rounds,
        surrogate_gradient_evaluated=surrogate_gradient is not None,
        surrogate_gradient_valid=bool(surrogate_valid),
        exact_objective=float(objective),
    )


def adaptive_hybrid_direction(
    *,
    physics: Helmholtz2D,
    surrogate: WavefieldSurrogate,
    m: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
    stability_mode: StabilityMode = "directional",
    max_exact_blocks: int | None = None,
    bound_scales: dict[str, float] | None = None,
) -> AdaptiveDirectionResult:
    """Build a descent direction using greedy exact-block repairs.

    The algorithm begins with the summed neural gradient. For the *current*
    hybrid direction it evaluates every unrepaired source/frequency block. If
    the sum of exact repaired contributions plus the selected upper bounds is
    negative, the direction passes the gate. With scales equal to one these
    are deterministic discrete bounds; calibrated scales have separate
    statistical semantics and require a valid calibration protocol.

    Otherwise the block with the largest current certificate uncertainty is
    replaced by its exact PDE gradient, the direction is recomputed, and all
    remaining certificates are recomputed for that new direction. This detail
    is essential: changing the hybrid gradient changes the direction being
    certified.

    Analytical stability is checked before neural inference or autodiff. If
    unavailable, the policy directly uses exact physics, or abstains when the
    exact budget is insufficient. Its unevaluated neural norm is recorded as
    zero with ``surrogate_gradient_evaluated=False``; no neural zero-gradient
    or stationarity claim is implied.

    A zero or nonfinite neural/partial hybrid gradient also causes full exact
    recovery when the total budget permits it, reusing previously paid blocks.
    Otherwise the method abstains without claiming true stationarity. This
    recovery accepts no newly constructed, partially calibrated repair path.
    """

    m = physics.validate_model(m)
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one survey block is required")
    max_exact = len(blocks) if max_exact_blocks is None else int(max_exact_blocks)
    if max_exact < 0:
        raise ValueError("max_exact_blocks must be non-negative")
    if bound_scales is not None:
        bound_scales = {label: _validate_scale(value) for label, value in bound_scales.items()}
    constants = _frequency_constants(
        physics,
        m,
        receiver_indices,
        [block.frequency_hz for block in blocks],
        need_beta=stability_mode in {"beta", "receiver"},
        need_receiver=stability_mode in {"receiver", "directional"},
        analytic_beta=stability_mode == "analytic",
    )
    if stability_mode == "analytic" and any(float(c["beta"]) <= 0 for c in constants.values()):
        if max_exact < len(blocks):
            return AdaptiveDirectionResult(
                mode="unavailable-stability-budget-exhausted", certified_descent=False,
                direction=np.zeros_like(m), repaired_blocks=(),
                exact_block_evaluations=0, total_blocks=len(blocks),
                initial_upper_derivative=float("inf"), final_upper_derivative=float("inf"),
                surrogate_gradient_norm=0.0, final_hybrid_gradient_norm=0.0,
                certificate_rounds=0, surrogate_gradient_evaluated=False,
                surrogate_gradient_valid=False,
            )
        return _exact_fallback(
            physics=physics, surrogate_gradient=None, m=m, blocks=blocks,
            receiver_indices=receiver_indices, mode="unavailable-stability-exact-fallback",
        )

    states: list[_BlockState] = []
    for block in blocks:
        q = physics.source(block.source_index)
        _, ng, _, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz,
        )
        states.append(_BlockState(block=block, q=q, neural_gradient=np.asarray(ng, dtype=float)))

    surrogate_gradient = np.sum([s.neural_gradient for s in states], axis=0)
    surrogate_norm = float(np.linalg.norm(surrogate_gradient))
    surrogate_valid = bool(np.isfinite(surrogate_norm))

    repaired: list[str] = []
    rounds = 0
    initial_upper = float("inf")

    while True:
        hybrid_gradient = np.sum(
            [s.exact_gradient if s.exact_gradient is not None else s.neural_gradient for s in states],
            axis=0,
        )
        hybrid_norm = float(np.linalg.norm(hybrid_gradient))
        exact_objective = (
            float(sum(state.exact_objective for state in states))
            if all(state.exact_objective is not None for state in states) else None
        )
        if not np.isfinite(hybrid_norm) or hybrid_norm <= 1e-14:
            # A zero/invalid neural or partial hybrid gradient says nothing
            # about stationarity of the true objective. Full exact recovery is
            # independent of calibrated bounds; do not start a different
            # partially repaired path under the old pathwise scale.
            if len(repaired) < len(states) and max_exact >= len(states):
                reason = "invalid-hybrid-gradient" if not np.isfinite(hybrid_norm) else "zero-hybrid-gradient"
                return _exact_fallback(
                    physics=physics, surrogate_gradient=surrogate_gradient, m=m,
                    blocks=blocks, receiver_indices=receiver_indices,
                    mode=f"{reason}-exact-fallback", initial_upper=initial_upper,
                    certificate_rounds=rounds, states=states,
                )
            all_exact = len(repaired) == len(states)
            if all_exact and not np.isfinite(hybrid_norm):
                raise ValueError("exact hybrid gradient norm must be finite")
            reason = "invalid-hybrid-gradient" if not np.isfinite(hybrid_norm) else "zero-hybrid-gradient"
            return AdaptiveDirectionResult(
                mode="stationary" if all_exact else f"{reason}-budget-exhausted",
                certified_descent=False,
                direction=np.zeros_like(m),
                repaired_blocks=tuple(repaired),
                exact_block_evaluations=len(repaired),
                total_blocks=len(states),
                initial_upper_derivative=initial_upper,
                final_upper_derivative=0.0 if all_exact else float("inf"),
                surrogate_gradient_norm=surrogate_norm if surrogate_valid else 0.0,
                final_hybrid_gradient_norm=hybrid_norm if np.isfinite(hybrid_norm) else 0.0,
                certificate_rounds=rounds,
                surrogate_gradient_valid=surrogate_valid,
                exact_objective=exact_objective,
            )

        direction = _normalised_negative(hybrid_gradient)

        rounds += 1
        contributions: list[tuple[_BlockState, float, float]] = []
        total_upper = 0.0
        # kappa depends on frequency and the current direction, not source.
        # It must be recomputed after repair changes the hybrid direction.
        kappas: dict[float, float] = {}
        for state in states:
            if state.exact_gradient is not None:
                exact_dd = float(np.dot(state.exact_gradient, direction))
                total_upper += exact_dd
                contributions.append((state, exact_dd, 0.0))
                continue

            frequency = float(state.block.frequency_hz)
            c = constants[frequency]
            kappa = None
            if stability_mode == "directional":
                if frequency not in kappas:
                    kappas[frequency] = physics.directional_receiver_tangent_resolvent_norm(
                        m, frequency, receiver_indices, direction,
                        receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
                    )
                kappa = kappas[frequency]
            cert = certify_direction(
                physics=physics,
                surrogate=surrogate,
                m=m,
                q=state.q,
                observed=state.block.observed,
                receiver_indices=receiver_indices,
                frequency_hz=frequency,
                direction=direction,
                beta=None if stability_mode == "directional" else float(c["beta"]),
                stability_mode=stability_mode,
                receiver_resolvent_matrix=c.get("receiver_matrix"),
                receiver_resolvent_norm=c.get("alpha"),
                directional_tangent_resolvent_norm=kappa,
            )
            scale = 1.0 if bound_scales is None else float(bound_scales.get(state.block.label, 1.0))
            scaled_bound = scale * cert.directional_error_bound
            upper = cert.surrogate_directional_derivative + scaled_bound
            total_upper += upper
            contributions.append((state, upper, scaled_bound))

        if not np.isfinite(initial_upper):
            initial_upper = float(total_upper)
        if total_upper < 0.0:
            mode = "neural" if not repaired else "selective-repair"
            if len(repaired) == len(states):
                mode = "exact-fallback"
            return AdaptiveDirectionResult(
                mode=mode,
                certified_descent=True,
                direction=direction,
                repaired_blocks=tuple(repaired),
                exact_block_evaluations=len(repaired),
                total_blocks=len(states),
                initial_upper_derivative=initial_upper,
                final_upper_derivative=float(total_upper),
                surrogate_gradient_norm=surrogate_norm,
                final_hybrid_gradient_norm=hybrid_norm,
                certificate_rounds=rounds,
                exact_objective=exact_objective,
            )

        unrepaired = [(s, bound) for s, _, bound in contributions if s.exact_gradient is None]
        if not unrepaired or len(repaired) >= max_exact:
            return AdaptiveDirectionResult(
                mode="budget-exhausted" if unrepaired else "exact-fallback",
                certified_descent=not unrepaired and total_upper < 0.0,
                direction=direction,
                repaired_blocks=tuple(repaired),
                exact_block_evaluations=len(repaired),
                total_blocks=len(states),
                initial_upper_derivative=initial_upper,
                final_upper_derivative=float(total_upper),
                surrogate_gradient_norm=surrogate_norm,
                final_hybrid_gradient_norm=hybrid_norm,
                certificate_rounds=rounds,
                exact_objective=exact_objective,
            )

        # Greedy repair: the largest uncertainty contributes the most avoidable
        # slack to the global upper bound.
        selected = max(unrepaired, key=lambda item: item[1])[0]
        exact_value, exact_gradient, _, _ = physics.objective_and_gradient(
            m,
            selected.q,
            selected.block.observed,
            receiver_indices,
            selected.block.frequency_hz,
        )
        selected.exact_gradient = np.asarray(exact_gradient, dtype=float)
        selected.exact_objective = float(exact_value)
        repaired.append(selected.block.label)


def global_fallback_direction(
    *,
    physics: Helmholtz2D,
    surrogate: WavefieldSurrogate,
    m: Array,
    blocks: Iterable[SurveyBlock],
    receiver_indices: tuple[int, ...],
    stability_mode: StabilityMode = "directional",
    bound_scale: float = 1.0,
) -> AdaptiveDirectionResult:
    """Certify the neural direction globally; otherwise use the exact gradient.

    Unavailable analytical stability is rejected before any neural work. The
    result records ``surrogate_gradient_evaluated=False`` and a zero norm
    sentinel when the neural gradient was deliberately not computed.
    """

    m = physics.validate_model(m)
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one survey block is required")
    bound_scale = _validate_scale(bound_scale)
    constants = _frequency_constants(
        physics,
        m,
        receiver_indices,
        [b.frequency_hz for b in blocks],
        need_beta=stability_mode in {"beta", "receiver"},
        need_receiver=stability_mode in {"receiver", "directional"},
        analytic_beta=stability_mode == "analytic",
    )
    if stability_mode == "analytic" and any(float(c["beta"]) <= 0 for c in constants.values()):
        return _exact_fallback(
            physics=physics, surrogate_gradient=None, m=m, blocks=blocks,
            receiver_indices=receiver_indices, mode="unavailable-stability-exact-fallback",
        )

    neural_grads: list[Array] = []
    for block in blocks:
        q = physics.source(block.source_index)
        _, ng, _, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        neural_grads.append(np.asarray(ng, dtype=float))
    neural_gradient = np.sum(neural_grads, axis=0)
    neural_norm = float(np.linalg.norm(neural_gradient))
    if not np.isfinite(neural_norm) or neural_norm <= 1e-14:
        reason = "invalid-hybrid-gradient" if not np.isfinite(neural_norm) else "zero-hybrid-gradient"
        return _exact_fallback(
            physics=physics, surrogate_gradient=neural_gradient, m=m, blocks=blocks,
            receiver_indices=receiver_indices, mode=f"{reason}-exact-fallback",
        )
    direction = _normalised_negative(neural_gradient)

    total_upper = 0.0
    kappas: dict[float, float] = {}
    for block in blocks:
        q = physics.source(block.source_index)
        c = constants[float(block.frequency_hz)]
        kappa = None
        if stability_mode == "directional":
            frequency = float(block.frequency_hz)
            if frequency not in kappas:
                kappas[frequency] = physics.directional_receiver_tangent_resolvent_norm(
                    m, frequency, receiver_indices, direction,
                    receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
                )
            kappa = kappas[frequency]
        cert = certify_direction(
            physics=physics,
            surrogate=surrogate,
            m=m,
            q=q,
            observed=block.observed,
            receiver_indices=receiver_indices,
            frequency_hz=block.frequency_hz,
            direction=direction,
            beta=None if stability_mode == "directional" else float(c["beta"]),
            stability_mode=stability_mode,
            receiver_resolvent_matrix=c.get("receiver_matrix"),
            receiver_resolvent_norm=c.get("alpha"),
            directional_tangent_resolvent_norm=kappa,
        )
        total_upper += cert.surrogate_directional_derivative + float(bound_scale) * cert.directional_error_bound

    if total_upper < 0.0:
        return AdaptiveDirectionResult(
            mode="neural",
            certified_descent=True,
            direction=direction,
            repaired_blocks=(),
            exact_block_evaluations=0,
            total_blocks=len(blocks),
            initial_upper_derivative=float(total_upper),
            final_upper_derivative=float(total_upper),
            surrogate_gradient_norm=float(np.linalg.norm(neural_gradient)),
            final_hybrid_gradient_norm=float(np.linalg.norm(neural_gradient)),
            certificate_rounds=1,
        )

    return _exact_fallback(
        physics=physics, surrogate_gradient=neural_gradient, m=m, blocks=blocks,
        receiver_indices=receiver_indices, mode="exact-fallback",
        initial_upper=float(total_upper), certificate_rounds=1,
    )
