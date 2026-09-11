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


def _normalised_negative(gradient: Array) -> Array:
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-14:
        return np.zeros_like(np.asarray(gradient, dtype=float))
    return -np.asarray(gradient, dtype=float) / norm


def _frequency_constants(
    physics: Helmholtz2D,
    m: Array,
    receiver_indices: tuple[int, ...],
    frequencies: Iterable[float],
    *,
    need_beta: bool = False,
) -> dict[float, dict[str, object]]:
    constants: dict[float, dict[str, object]] = {}
    for frequency in sorted({float(v) for v in frequencies}):
        receiver_matrix = physics.receiver_resolvent_matrix(
            m, frequency, receiver_indices
        )
        entry: dict[str, object] = {
            "receiver_matrix": receiver_matrix,
            "alpha": physics.receiver_resolvent_norm(
                m,
                frequency,
                receiver_indices,
                receiver_resolvent_matrix=receiver_matrix,
            ),
        }
        if need_beta:
            entry["beta"] = physics.smallest_singular_value(m, frequency)
        constants[frequency] = entry
    return constants


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
    """Build a descent direction with the minimum greedy exact-block repairs.

    The algorithm begins with the summed neural gradient. For the *current*
    hybrid direction it certifies every unrepaired source/frequency block. If
    the sum of exact repaired contributions plus rigorous upper bounds for the
    unrepaired contributions is negative, the direction is certified.

    Otherwise the block with the largest current certificate uncertainty is
    replaced by its exact PDE gradient, the direction is recomputed, and all
    remaining certificates are recomputed for that new direction. This detail
    is essential: changing the hybrid gradient changes the direction being
    certified.
    """

    m = physics.validate_model(m)
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one survey block is required")
    max_exact = len(blocks) if max_exact_blocks is None else int(max_exact_blocks)
    if max_exact < 0:
        raise ValueError("max_exact_blocks must be non-negative")

    states: list[_BlockState] = []
    for block in blocks:
        q = physics.source(block.source_index)
        _, ng, _, _ = surrogate.objective_and_gradient(
            m,
            q,
            block.observed,
            receiver_indices,
            block.frequency_hz,
        )
        states.append(_BlockState(block=block, q=q, neural_gradient=np.asarray(ng, dtype=float)))

    surrogate_gradient = np.sum([s.neural_gradient for s in states], axis=0)
    surrogate_norm = float(np.linalg.norm(surrogate_gradient))
    constants = _frequency_constants(
        physics,
        m,
        receiver_indices,
        [s.block.frequency_hz for s in states],
        need_beta=stability_mode != "directional",
    )

    repaired: list[str] = []
    rounds = 0
    initial_upper = float("nan")

    while True:
        hybrid_gradient = np.sum(
            [s.exact_gradient if s.exact_gradient is not None else s.neural_gradient for s in states],
            axis=0,
        )
        direction = _normalised_negative(hybrid_gradient)
        hybrid_norm = float(np.linalg.norm(hybrid_gradient))
        if np.linalg.norm(direction) <= 1e-14:
            return AdaptiveDirectionResult(
                mode="stationary",
                certified_descent=False,
                direction=direction,
                repaired_blocks=tuple(repaired),
                exact_block_evaluations=len(repaired),
                total_blocks=len(states),
                initial_upper_derivative=float("nan"),
                final_upper_derivative=0.0,
                surrogate_gradient_norm=surrogate_norm,
                final_hybrid_gradient_norm=hybrid_norm,
                certificate_rounds=rounds,
            )

        rounds += 1
        contributions: list[tuple[_BlockState, float, float]] = []
        total_upper = 0.0
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
                kappa = physics.directional_receiver_tangent_resolvent_norm(
                    m,
                    frequency,
                    receiver_indices,
                    direction,
                    receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
                )
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
                receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
                receiver_resolvent_norm=float(c["alpha"]),
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
            )

        # Greedy repair: the largest uncertainty contributes the most avoidable
        # slack to the global upper bound.
        selected = max(unrepaired, key=lambda item: item[1])[0]
        _, exact_gradient, _, _ = physics.objective_and_gradient(
            m,
            selected.q,
            selected.block.observed,
            receiver_indices,
            selected.block.frequency_hz,
        )
        selected.exact_gradient = np.asarray(exact_gradient, dtype=float)
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
    """Certify the neural direction globally; otherwise use the exact gradient."""

    m = physics.validate_model(m)
    blocks = tuple(blocks)
    neural_grads: list[Array] = []
    for block in blocks:
        q = physics.source(block.source_index)
        _, ng, _, _ = surrogate.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        neural_grads.append(np.asarray(ng, dtype=float))
    neural_gradient = np.sum(neural_grads, axis=0)
    direction = _normalised_negative(neural_gradient)
    constants = _frequency_constants(
        physics,
        m,
        receiver_indices,
        [b.frequency_hz for b in blocks],
        need_beta=stability_mode != "directional",
    )

    total_upper = 0.0
    for block in blocks:
        q = physics.source(block.source_index)
        c = constants[float(block.frequency_hz)]
        kappa = None
        if stability_mode == "directional":
            kappa = physics.directional_receiver_tangent_resolvent_norm(
                m,
                block.frequency_hz,
                receiver_indices,
                direction,
                receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
            )
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
            receiver_resolvent_matrix=np.asarray(c["receiver_matrix"]),
            receiver_resolvent_norm=float(c["alpha"]),
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

    exact_grads: list[Array] = []
    for block in blocks:
        q = physics.source(block.source_index)
        _, eg, _, _ = physics.objective_and_gradient(
            m, q, block.observed, receiver_indices, block.frequency_hz
        )
        exact_grads.append(np.asarray(eg, dtype=float))
    exact_gradient = np.sum(exact_grads, axis=0)
    exact_direction = _normalised_negative(exact_gradient)
    exact_dd = -float(np.linalg.norm(exact_gradient))
    return AdaptiveDirectionResult(
        mode="exact-fallback",
        certified_descent=exact_dd < 0.0,
        direction=exact_direction,
        repaired_blocks=tuple(b.label for b in blocks),
        exact_block_evaluations=len(blocks),
        total_blocks=len(blocks),
        initial_upper_derivative=float(total_upper),
        final_upper_derivative=exact_dd,
        surrogate_gradient_norm=float(np.linalg.norm(neural_gradient)),
        final_hybrid_gradient_norm=float(np.linalg.norm(exact_gradient)),
        certificate_rounds=1,
    )
