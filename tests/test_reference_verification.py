import numpy as np

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.physics.verification import verify_reference_problem


def test_reference_passes_dot_gradient_and_taylor_tests():
    problem = build_synthetic_problem(
        shape=(10, 10), frequencies_hz=(3.0,), n_sources=1, n_receivers=7
    )
    block = problem.blocks[0]
    q = problem.physics.source(block.source_index)
    report = verify_reference_problem(
        problem.physics,
        problem.m0,
        q,
        block.observed,
        problem.geometry.receiver_indices,
        block.frequency_hz,
        seed=9,
        n_trials=3,
    )
    d = report.to_dict()
    assert d["max_dot_relative_error"] < 1e-8
    assert d["max_gradient_relative_error"] < 5e-4
    assert d["min_taylor_slope"] > 1.8
    assert np.isfinite(d["min_taylor_slope"])
