import numpy as np

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_surrogate_gradient_matches_finite_difference():
    problem = build_synthetic_problem(shape=(9, 9), frequencies_hz=(3.0,), n_sources=1, n_receivers=5)
    block = problem.blocks[0]
    q = problem.physics.source(block.source_index)
    surrogate = SmoothedHelmholtzSurrogate(problem.physics, smoothing_sigma=0.7, model_scale=1.01)
    _, g, _, _ = surrogate.objective_and_gradient(
        problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    rng = np.random.default_rng(4)
    v = rng.normal(size=problem.physics.n)
    v /= np.linalg.norm(v)
    eps = 1e-6
    fp, _, _, _ = surrogate.objective_and_gradient(
        problem.m0 + eps * v, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    fm, _, _, _ = surrogate.objective_and_gradient(
        problem.m0 - eps * v, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    fd = (fp - fm) / (2 * eps)
    assert np.isclose(fd, float(np.dot(g, v)), rtol=5e-4, atol=1e-7)
