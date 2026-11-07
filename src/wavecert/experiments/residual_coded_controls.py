"""Operation-count and optional timing controls for the exact receiver-coded gradient."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.physics.receiver_oracle import receiver_map_directional_oracle
from wavecert.physics.residual_coded import exact_survey_objective_gradient


def _rhs(cost):
    return sum(value for name, value in cost.items() if name.endswith("_rhs"))


def measure_residual_coded_controls(
    *, sources: int, receivers: int, shape: tuple[int, int] = (32, 32),
    frequencies_hz: tuple[float, ...] = (3.0, 4.0), seed: int = 20261107,
    timing_repeats: int = 0,
) -> dict:
    """Measure executed work without timing by default.

    Positive timing_repeats enables repeated cold-solver invocation timing;
    these are local descriptive timings, not a full inversion comparison.
    Data synthesis, scalar reference labels, and finite differences use
    separate solvers and are excluded from each control's measured work.
    """
    for name, value in (("sources", sources), ("receivers", receivers), ("timing_repeats", timing_repeats)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or value < (0 if name == "timing_repeats" else 1)):
            raise ValueError(f"{name} has an invalid count")
    if (not frequencies_hz or len(set(frequencies_hz)) != len(frequencies_hz)
            or any(not np.isfinite(f) or f <= 0 for f in frequencies_hz)):
        raise ValueError("frequencies must be finite, positive, unique, and nonempty")
    p = Helmholtz2D(shape=shape, spacing=0.05, damping_width=3, damping_strength=2.0)
    geometry = SurveyGeometry.from_grid(shape, n_sources=sources, n_receivers=receivers,
        source_depth=2, receiver_depth=2, margin=2)
    if (len(set(geometry.source_indices)) != sources
            or len(set(geometry.receiver_indices)) != receivers):
        raise ValueError("use distinct source/receiver positions in the geometry sweep")
    rng = np.random.default_rng(seed)
    model = rng.uniform(0.18, 0.27, p.n)
    direction = rng.normal(size=p.n)
    direction /= np.linalg.norm(direction)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, sources, receivers]))
    blocks = []
    for frequency in frequencies_hz:
        for source in geometry.source_indices:
            current = p.restrict(p.solve_state(model, p.source(source), frequency), geometry.receiver_indices)
            noise = noise_rng.normal(size=receivers)+1j*noise_rng.normal(size=receivers)
            noise *= 0.05*np.linalg.norm(current)/np.linalg.norm(noise)
            blocks.append(SurveyBlock(int(source), frequency, current+noise, f"{source}@{frequency}"))
    blocks = tuple(blocks)
    covered = len(set(geometry.source_indices) & set(geometry.receiver_indices))
    f = len(frequencies_hz)
    expected = {"ordinary-gradient": 2*sources*f, "symmetry-gradient": (receivers+sources-covered)*f,
        "coded-gradient": 2*receivers*f,
        "best-gradient": min(2*sources, receivers+sources-covered, 2*receivers)*f,
        "exact-directional-maps": 2*receivers*f, "norm-only-initial-maps": 2*receivers*f}

    # Keep the established scalar implementation as an independent reference
    # for the new batched ordinary, symmetry, and coded control routes.
    reference_solver = replace(p)
    reference_objective, reference_gradient = 0.0, np.zeros(p.n)
    for block in blocks:
        objective, gradient, _, _ = reference_solver.objective_and_gradient(model,
            reference_solver.source(block.source_index), block.observed,
            geometry.receiver_indices, block.frequency_hz)
        reference_objective += objective
        reference_gradient += gradient
    reference_dd = float(np.dot(reference_gradient, direction))

    def gradient_method(physics, strategy):
        result = exact_survey_objective_gradient(physics=physics, m=model, blocks=blocks,
            receiver_indices=geometry.receiver_indices, strategy=strategy)
        return result.objective, float(np.dot(result.gradient, direction)), result.gradient, result.frequency_strategies

    def directional(physics):
        objective, dd = receiver_map_directional_oracle(physics=physics, m=model, blocks=blocks,
            receiver_indices=geometry.receiver_indices, direction=direction)
        return objective, dd, None, {}

    def norms(physics):
        for frequency in frequencies_hz:
            matrix = physics.receiver_resolvent_matrix(model, frequency, geometry.receiver_indices)
            physics.receiver_resolvent_norm(model, frequency, geometry.receiver_indices,
                receiver_resolvent_matrix=matrix)
            physics.directional_receiver_tangent_resolvent_norm(model, frequency, geometry.receiver_indices,
                direction, receiver_resolvent_matrix=matrix)
        return None, None, None, {}

    methods = {"ordinary-gradient": lambda physics: gradient_method(physics, "ordinary"),
        "symmetry-gradient": lambda physics: gradient_method(physics, "symmetry"),
        "coded-gradient": lambda physics: gradient_method(physics, "coded"),
        "best-gradient": lambda physics: gradient_method(physics, "best"),
        "exact-directional-maps": directional, "norm-only-initial-maps": norms}
    collected = {name: [] for name in methods}
    names = list(methods)
    for repeat in range(max(1, timing_repeats)):
        order = names[repeat % len(names):]+names[:repeat % len(names)]
        for name in order:
            physics = replace(p)
            start = perf_counter() if timing_repeats else None
            objective, dd, gradient, strategies = methods[name](physics)
            elapsed = perf_counter()-start if start is not None else None
            costs = physics.cost_snapshot()
            if (_rhs(costs) != expected[name] or costs["factorizations"] != f
                    or costs["svd_evaluations"] != (2*f if name == "norm-only-initial-maps" else 0)):
                raise AssertionError(f"executed work differs from prediction for {name}: {costs}")
            errors = {"objective_absolute_error": None, "directional_absolute_error": None,
                "gradient_relative_l2_error": None}
            if objective is not None:
                errors["objective_absolute_error"] = float(abs(objective-reference_objective))
                errors["directional_absolute_error"] = float(abs(dd-reference_dd))
                if (not np.isclose(objective, reference_objective, rtol=1e-9, atol=1e-11)
                        or not np.isclose(dd, reference_dd, rtol=1e-9, atol=1e-11)):
                    raise AssertionError("exact control objective/directional disagreement")
            if gradient is not None:
                errors["gradient_relative_l2_error"] = float(np.linalg.norm(gradient-reference_gradient)
                    / max(np.linalg.norm(reference_gradient), 1e-15))
                if errors["gradient_relative_l2_error"] > 1e-9:
                    raise AssertionError("exact control gradient disagreement")
            collected[name].append({"costs": costs, "elapsed_seconds": elapsed, "objective": objective,
                "directional_derivative": dd, "full_gradient_constructed": gradient is not None,
                "frequency_strategies": {str(k): v for k, v in strategies.items()}, **errors})

    finite_difference_solver = replace(p)

    def phi(m):
        return sum(finite_difference_solver.objective(m, finite_difference_solver.source(b.source_index),
            b.observed, geometry.receiver_indices, b.frequency_hz) for b in blocks)

    finite_differences = []
    for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        difference = (phi(model+step*direction)-phi(model-step*direction))/(2*step)
        finite_differences.append({"step": step, "directional_derivative": float(difference),
            "relative_error": float(abs(difference-reference_dd)/max(abs(reference_dd), 1e-15))})
    best_fd = min(point["relative_error"] for point in finite_differences)
    if best_fd > 1e-5:
        raise AssertionError(f"independent objective finite difference failed: {best_fd}")
    rows = []
    for name, observations in collected.items():
        errors = {key: None if observations[0][key] is None else max(o[key] for o in observations)
            for key in ("objective_absolute_error", "directional_absolute_error", "gradient_relative_l2_error")}
        rows.append({"method": name, "expected_rhs": expected[name], "executed_rhs": _rhs(observations[0]["costs"]),
            "costs": observations[0]["costs"], "full_gradient_constructed": observations[0]["full_gradient_constructed"],
            "frequency_strategies": observations[0]["frequency_strategies"],
            "elapsed_seconds": [o["elapsed_seconds"] for o in observations] if timing_repeats else [],
            "median_elapsed_seconds": float(np.median([o["elapsed_seconds"] for o in observations])) if timing_repeats else None,
            "all_repetitions": observations, **errors})
    if expected["best-gradient"] > expected["exact-directional-maps"]:
        raise AssertionError("best exact route must include the 2r coded option")
    return {"sources": sources, "receivers": receivers, "covered_sources": covered,
        "shape": list(shape), "spacing": p.spacing, "frequencies_hz": list(frequencies_hz),
        "seed": seed, "timing_repeats": timing_repeats, "all_checks_pass": True,
        "reference_objective": float(reference_objective), "reference_directional_derivative": reference_dd,
        "reference_gradient_norm": float(np.linalg.norm(reference_gradient)),
        "scalar_reference_validation_costs_excluded_from_controls": reference_solver.cost_snapshot(),
        "finite_difference_validation_costs_excluded_from_controls": finite_difference_solver.cost_snapshot(),
        "finite_difference_best_relative_error": best_fd, "finite_difference_points": finite_differences,
        "best_rhs_no_greater_than_initial_map_rhs": True, "methods": rows}
