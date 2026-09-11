import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_directional_certificate_covers_realized_error():
    problem = build_synthetic_problem(shape=(9, 9), frequencies_hz=(3.0,), n_sources=1, n_receivers=5)
    block = problem.blocks[0]
    physics = problem.physics
    surrogate = SmoothedHelmholtzSurrogate(physics, smoothing_sigma=0.8, model_scale=1.015)
    q = physics.source(block.source_index)
    rng = np.random.default_rng(9)

    beta = physics.smallest_singular_value(problem.m0, block.frequency_hz)
    for _ in range(4):
        v = rng.normal(size=physics.n)
        v /= np.linalg.norm(v)
        cert = certify_direction(
            physics=physics,
            surrogate=surrogate,
            m=problem.m0,
            q=q,
            observed=block.observed,
            receiver_indices=problem.geometry.receiver_indices,
            frequency_hz=block.frequency_hz,
            direction=v,
            beta=beta,
            validate_exact=True,
        )
        assert cert.realized_error is not None
        assert cert.directional_error_bound + 1e-9 >= cert.realized_error
        assert cert.state_error_bound >= 0
        assert cert.tangent_error_bound >= 0


def test_certificate_flag_implies_true_descent():
    problem = build_synthetic_problem(shape=(9, 9), frequencies_hz=(3.0,), n_sources=1, n_receivers=5)
    block = problem.blocks[0]
    physics = problem.physics
    surrogate = SmoothedHelmholtzSurrogate(physics, smoothing_sigma=0.4, model_scale=1.003)
    q = physics.source(block.source_index)
    _, sg, _, _ = surrogate.objective_and_gradient(
        problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    v = -sg / np.linalg.norm(sg)
    cert = certify_direction(
        physics=physics,
        surrogate=surrogate,
        m=problem.m0,
        q=q,
        observed=block.observed,
        receiver_indices=problem.geometry.receiver_indices,
        frequency_hz=block.frequency_hz,
        direction=v,
        validate_exact=True,
    )
    if cert.certified_descent:
        assert cert.exact_directional_derivative is not None
        assert cert.exact_directional_derivative < 0
