import numpy as np

from wavecert.experiments.synthetic import build_synthetic_problem


def test_exact_gradient_matches_finite_difference():
    problem = build_synthetic_problem(shape=(10, 10), frequencies_hz=(3.0,), n_sources=1, n_receivers=6)
    block = problem.blocks[0]
    q = problem.physics.source(block.source_index)
    f0, g, _, _ = problem.physics.objective_and_gradient(
        problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    rng = np.random.default_rng(3)
    v = rng.normal(size=problem.physics.n)
    v /= np.linalg.norm(v)

    eps = 1e-6
    fp, _, _, _ = problem.physics.objective_and_gradient(
        problem.m0 + eps * v, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    fm, _, _, _ = problem.physics.objective_and_gradient(
        problem.m0 - eps * v, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    fd = (fp - fm) / (2 * eps)
    ad = float(np.dot(g, v))
    assert np.isfinite(f0)
    assert np.isclose(fd, ad, rtol=3e-4, atol=1e-7)
