import numpy as np

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.optimization.fwi import exact_objective, exact_objective_gradient
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_exact_objective_gradient_and_short_inversion_are_finite():
    problem = build_synthetic_problem(shape=(10, 10), frequencies_hz=(3.0,), n_sources=1, n_receivers=8)
    value, gradient = exact_objective_gradient(
        problem.physics, problem.m0, problem.blocks, problem.geometry.receiver_indices
    )
    direct = exact_objective(
        problem.physics, problem.m0, problem.blocks, problem.geometry.receiver_indices
    )
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    assert np.isclose(value, direct)

    # Smoke-test surrogate compatibility used by the Phase-7 optimizer.
    surrogate = SmoothedHelmholtzSurrogate(problem.physics, smoothing_sigma=0.5)
    value_s, grad_s, _, _ = surrogate.objective_and_gradient(
        problem.m0,
        problem.physics.source(problem.blocks[0].source_index),
        problem.blocks[0].observed,
        problem.geometry.receiver_indices,
        problem.blocks[0].frequency_hz,
    )
    assert np.isfinite(value_s)
    assert np.isfinite(grad_s).all()
