import numpy as np

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.repair.adaptive import adaptive_hybrid_direction
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_rigorous_adaptive_repair_returns_true_descent_direction():
    problem = build_synthetic_problem(
        shape=(8, 8), frequencies_hz=(3.0, 3.5), n_sources=2, n_receivers=6
    )
    surrogate = SmoothedHelmholtzSurrogate(
        problem.physics, smoothing_sigma=0.8, model_scale=1.015
    )
    result = adaptive_hybrid_direction(
        physics=problem.physics,
        surrogate=surrogate,
        m=problem.m0,
        blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices,
        stability_mode="directional",
    )
    exact_gradient = np.zeros(problem.physics.n)
    for block in problem.blocks:
        q = problem.physics.source(block.source_index)
        _, g, _, _ = problem.physics.objective_and_gradient(
            problem.m0,
            q,
            block.observed,
            problem.geometry.receiver_indices,
            block.frequency_hz,
        )
        exact_gradient += g
    assert result.certified_descent
    assert float(np.dot(exact_gradient, result.direction)) < 0.0
    assert result.exact_block_evaluations <= len(problem.blocks)
