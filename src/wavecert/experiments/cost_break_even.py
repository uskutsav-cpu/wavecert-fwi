"""Engineering operation-count audit of the exact receiver-map certificate.

This deliberately measures RHS counts rather than timing or neural-model
quality. The count comparison applies to the current ``directional`` reference
implementation, not to all possible certification methods.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry


def receiver_map_cost_model(*, sources: int, receivers: int, frequencies: int) -> dict:
    """Compare initial map/kappa RHS work with one complete exact gradient.

    Each receiver contributes one RHS for P A^-1 and one for the directional
    map P A^-1 diag(v) A^-1. Each source contributes one exact forward and one
    exact adjoint RHS. Frequency factorizations are reusable in both paths.
    A lower setup RHS count is necessary, not sufficient, for an RHS saving.
    It is not a necessary or sufficient condition for a wall-clock speedup.
    """
    for name, value in (("sources", sources), ("receivers", receivers), ("frequencies", frequencies)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    sources, receivers, frequencies = int(sources), int(receivers), int(frequencies)
    certificate_rhs = 2 * receivers * frequencies
    gradient_rhs = 2 * sources * frequencies
    return {
        "sources_per_frequency": sources,
        "receivers": receivers,
        "frequencies": frequencies,
        "initial_certificate_map_rhs": certificate_rhs,
        "complete_exact_gradient_rhs": gradient_rhs,
        "certificate_to_gradient_rhs_ratio": certificate_rhs / gradient_rhs,
        "initial_map_rhs_saving_possible": certificate_rhs < gradient_rhs,
        "rhs_saving_impossible_even_with_zero_repairs": certificate_rhs >= gradient_rhs,
    }


def measure_receiver_map_cost(
    *,
    sources: int,
    receivers: int,
    shape: tuple[int, int] = (32, 32),
    frequencies_hz: tuple[float, ...] = (3.0, 4.0),
    seed: int = 20261107,
) -> dict:
    """Execute one cold receiver-map setup and one cold full exact gradient.

    One fixed model/direction is regenerated from ``seed`` for every geometry.
    Observations are synthesized outside both measured branches, using exact
    current receiver values plus 5% complex noise. This ensures nonzero
    residuals and meaningful adjoint right-hand sides. No FNO is loaded.
    """
    if not frequencies_hz or len(set(frequencies_hz)) != len(frequencies_hz):
        raise ValueError("frequencies_hz must be nonempty and unique")
    predicted = receiver_map_cost_model(
        sources=sources, receivers=receivers, frequencies=len(frequencies_hz),
    )
    physics = Helmholtz2D(shape=shape, spacing=0.05, damping_width=3, damping_strength=2.0)
    geometry = SurveyGeometry.from_grid(
        shape, n_sources=sources, n_receivers=receivers, source_depth=2, receiver_depth=2, margin=2,
    )
    if len(set(geometry.source_indices)) != sources or len(set(geometry.receiver_indices)) != receivers:
        raise ValueError("requested geometry repeats positions; use a larger grid or fewer sources/receivers")
    rng = np.random.default_rng(seed)
    model = rng.uniform(0.18, 0.27, size=physics.n)
    direction = rng.normal(size=physics.n)
    direction /= np.linalg.norm(direction)

    # Use a third solver so data synthesis cannot warm either measured cache.
    observations = {}
    residual_norms = []
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, sources, receivers]))
    for frequency in frequencies_hz:
        for source in geometry.source_indices:
            state = physics.solve_state(model, physics.source(source), frequency)
            current = physics.restrict(state, geometry.receiver_indices)
            noise = noise_rng.normal(size=receivers) + 1j * noise_rng.normal(size=receivers)
            noise *= 0.05 * np.linalg.norm(current) / np.linalg.norm(noise)
            observations[source, frequency] = current + noise
            residual_norms.append(float(np.linalg.norm(noise)))

    certificate_physics = replace(physics)
    for frequency in frequencies_hz:
        matrix = certificate_physics.receiver_resolvent_matrix(model, frequency, geometry.receiver_indices)
        certificate_physics.receiver_resolvent_norm(
            model, frequency, geometry.receiver_indices, receiver_resolvent_matrix=matrix,
        )
        certificate_physics.directional_receiver_tangent_resolvent_norm(
            model, frequency, geometry.receiver_indices, direction, receiver_resolvent_matrix=matrix,
        )
    certificate_costs = certificate_physics.cost_snapshot()

    exact_physics = replace(physics)
    objectives = []
    for frequency in frequencies_hz:
        for source in geometry.source_indices:
            objective, gradient, _, _ = exact_physics.objective_and_gradient(
                model, exact_physics.source(source), observations[source, frequency],
                geometry.receiver_indices, frequency,
            )
            if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
                raise RuntimeError("reference objective or gradient is non-finite")
            objectives.append(float(objective))
    exact_costs = exact_physics.cost_snapshot()
    certificate_rhs = certificate_costs["stability_rhs"]
    exact_rhs = exact_costs["forward_rhs"] + exact_costs["adjoint_rhs"]
    counts_match = (
        certificate_rhs == predicted["initial_certificate_map_rhs"]
        and exact_rhs == predicted["complete_exact_gradient_rhs"]
        and certificate_costs["factorizations"] == len(frequencies_hz)
        and exact_costs["factorizations"] == len(frequencies_hz)
    )
    if not counts_match:
        raise AssertionError("executed costs differ from the stated implementation model")
    return {
        **predicted,
        "shape": list(shape), "spacing": physics.spacing, "frequencies_hz": list(frequencies_hz),
        "seed": seed, "minimum_synthetic_residual_norm": min(residual_norms),
        "exact_objective": float(sum(objectives)),
        "observed_initial_map_rhs": certificate_rhs,
        "observed_full_gradient_rhs": exact_rhs,
        "certificate_costs": certificate_costs,
        "exact_gradient_costs": exact_costs,
        "observed_counts_match_formulas": counts_match,
    }
