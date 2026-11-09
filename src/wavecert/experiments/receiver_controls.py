"""Cold-solver controls for exact gradients and receiver-map directional tests."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from wavecert.experiments.cost_break_even import receiver_map_cost_model
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.physics.receiver_oracle import (
    receiver_map_directional_oracle,
    reciprocity_objective_gradient,
)


def _rhs(cost: dict[str, int]) -> int:
    return sum(value for key, value in cost.items() if key.endswith("_rhs"))


def measure_receiver_controls(
    *, sources: int, receivers: int, shape: tuple[int, int] = (32, 32),
    frequencies_hz: tuple[float, ...] = (3.0, 4.0), seed: int = 20261107,
    repeats: int = 3,
) -> dict:
    """Compare four scoped controls on one seeded model and direction.

    Observations and finite differences use separate solvers outside all
    timed branches. Every timed repetition starts with a cold independent
    solver; method order rotates. Neural proposal generation, optimization,
    training and finite-step acceptance are absent. Timings are descriptive,
    not a claim of an inversion speedup. Only the two gradient controls
    construct a complete exact model-space gradient.
    """
    receiver_map_cost_model(sources=sources, receivers=receivers, frequencies=len(frequencies_hz))
    if (isinstance(repeats, (bool, np.bool_))
            or not isinstance(repeats, (int, np.integer)) or repeats < 1):
        raise ValueError("repeats must be a positive integer")
    if (not frequencies_hz or len(set(frequencies_hz)) != len(frequencies_hz)
            or any(not np.isfinite(f) or f <= 0 for f in frequencies_hz)):
        raise ValueError("frequencies must be positive, finite, nonempty, and unique")
    physics = Helmholtz2D(shape=shape, spacing=0.05, damping_width=3, damping_strength=2.0)
    geometry = SurveyGeometry.from_grid(
        shape, n_sources=sources, n_receivers=receivers, source_depth=2,
        receiver_depth=2, margin=2,
    )
    if (len(set(geometry.source_indices)) != sources
            or len(set(geometry.receiver_indices)) != receivers):
        raise ValueError("geometry must contain distinct source and receiver positions")
    rng = np.random.default_rng(seed)
    model = rng.uniform(0.18, 0.27, physics.n)
    direction = rng.normal(size=physics.n)
    direction /= np.linalg.norm(direction)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, sources, receivers]))
    blocks = []
    for frequency in frequencies_hz:
        for source in geometry.source_indices:
            current = physics.restrict(
                physics.solve_state(model, physics.source(source), frequency), geometry.receiver_indices,
            )
            noise = noise_rng.normal(size=receivers) + 1j * noise_rng.normal(size=receivers)
            noise *= 0.05 * np.linalg.norm(current) / np.linalg.norm(noise)
            blocks.append(SurveyBlock(source, frequency, current + noise, f"{source}@{frequency}"))
    blocks = tuple(blocks)
    covered = len(set(geometry.source_indices) & set(geometry.receiver_indices))
    f = len(frequencies_hz)
    expected = {
        "ordinary-exact-gradient": 2 * sources * f,
        "reciprocity-exact-gradient": min(2 * sources, receivers + sources - covered) * f,
        "receiver-map-exact-direction": 2 * receivers * f,
        "norm-only-initial-maps": 2 * receivers * f,
    }

    def ordinary(p):
        objective = 0.0
        gradient = np.zeros(p.n)
        for block in blocks:
            value, grad, _, _ = p.objective_and_gradient(
                model, p.source(block.source_index), block.observed,
                geometry.receiver_indices, block.frequency_hz,
            )
            objective += value
            gradient += grad
        return objective, float(np.dot(gradient, direction)), gradient, {}

    def reciprocal(p):
        result = reciprocity_objective_gradient(
            physics=p, m=model, blocks=blocks, receiver_indices=geometry.receiver_indices,
        )
        if result.expected_rhs != expected["reciprocity-exact-gradient"]:
            raise AssertionError("reciprocity strategy differs from predicted RHS work")
        return (result.objective, float(np.dot(result.gradient, direction)), result.gradient,
                {str(k): v for k, v in result.frequency_strategies.items()})

    def oracle(p):
        objective, derivative = receiver_map_directional_oracle(
            physics=p, m=model, blocks=blocks, receiver_indices=geometry.receiver_indices,
            direction=direction,
        )
        return objective, derivative, None, {}

    def norms(p):
        for frequency in frequencies_hz:
            matrix = p.receiver_resolvent_matrix(model, frequency, geometry.receiver_indices)
            p.receiver_resolvent_norm(
                model, frequency, geometry.receiver_indices, receiver_resolvent_matrix=matrix,
            )
            p.directional_receiver_tangent_resolvent_norm(
                model, frequency, geometry.receiver_indices, direction,
                receiver_resolvent_matrix=matrix,
            )
        return None, None, None, {}

    methods = {"ordinary-exact-gradient": ordinary, "reciprocity-exact-gradient": reciprocal,
               "receiver-map-exact-direction": oracle, "norm-only-initial-maps": norms}
    measured: dict[str, list[dict]] = {name: [] for name in methods}
    names = list(methods)
    for repetition in range(repeats):
        order = names[repetition % len(names):] + names[:repetition % len(names)]
        for name in order:
            p = replace(physics)
            start = perf_counter()
            objective, derivative, gradient, strategies = methods[name](p)
            elapsed = perf_counter() - start
            costs = p.cost_snapshot()
            if (_rhs(costs) != expected[name] or costs["factorizations"] != f
                    or costs["svd_evaluations"] != (2 * f if name == "norm-only-initial-maps" else 0)):
                raise AssertionError(f"unexpected executed work for {name}: {costs}")
            measured[name].append(dict(objective=objective, derivative=derivative, gradient=gradient,
                                       elapsed=elapsed, costs=costs, strategies=strategies))

    reference = measured["ordinary-exact-gradient"][0]
    rows = []
    for name, observations in measured.items():
        first = observations[0]
        max_objective_error = max_directional_error = max_gradient_error = None
        if first["objective"] is not None:
            max_objective_error = max(abs(o["objective"] - reference["objective"]) for o in observations)
            max_directional_error = max(abs(o["derivative"] - reference["derivative"]) for o in observations)
            for observation in observations:
                if (not np.isclose(observation["objective"], reference["objective"], rtol=1e-9, atol=1e-11)
                        or not np.isclose(observation["derivative"], reference["derivative"], rtol=1e-9, atol=1e-11)):
                    raise AssertionError("receiver control objective/directional disagreement")
        if first["gradient"] is not None:
            max_gradient_error = max(
                float(np.linalg.norm(o["gradient"] - reference["gradient"])) for o in observations
            )
            if any(not np.allclose(o["gradient"], reference["gradient"], rtol=1e-9, atol=1e-11)
                   for o in observations):
                raise AssertionError("reciprocity gradient disagreement")
        rows.append({
            "method": name, "expected_rhs": expected[name], "executed_rhs": _rhs(first["costs"]),
            "costs": first["costs"], "all_repetition_costs": [o["costs"] for o in observations],
            "elapsed_seconds": [o["elapsed"] for o in observations],
            "median_elapsed_seconds": float(np.median([o["elapsed"] for o in observations])),
            "objective": first["objective"], "directional_derivative": first["derivative"],
            "complete_gradient_constructed": first["gradient"] is not None,
            "maximum_objective_absolute_error": max_objective_error,
            "maximum_directional_absolute_error": max_directional_error,
            "maximum_gradient_l2_error": max_gradient_error,
            "frequency_strategies": first["strategies"],
        })

    fd_physics = replace(physics)

    def objective_at(m):
        return sum(fd_physics.objective(
            m, fd_physics.source(b.source_index), b.observed, geometry.receiver_indices, b.frequency_hz,
        ) for b in blocks)

    finite_differences = []
    for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        fd = (objective_at(model + step * direction) - objective_at(model - step * direction)) / (2 * step)
        finite_differences.append({"step": step, "derivative": float(fd),
            "relative_error": float(abs(fd - reference["derivative"]) / max(abs(reference["derivative"]), 1e-15))})
    best_fd_error = min(point["relative_error"] for point in finite_differences)
    if best_fd_error > 1e-5:
        raise AssertionError(f"independent finite difference failed: {best_fd_error}")
    return {
        "sources": sources, "receivers": receivers, "covered_sources": covered,
        "shape": list(shape), "spacing": physics.spacing, "frequencies_hz": list(frequencies_hz),
        "seed": seed, "repeats": repeats,
        "model_realizations": 1, "all_checks_pass": True,
        "finite_difference_best_relative_error": best_fd_error,
        "finite_difference_points": finite_differences, "methods": rows,
    }
