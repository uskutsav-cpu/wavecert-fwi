import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_receiver_resolvent_sparse_matches_dense_reference():
    problem = build_synthetic_problem(shape=(7, 7), frequencies_hz=(3.0,), n_sources=1, n_receivers=5)
    physics = problem.physics
    block = problem.blocks[0]
    inv_a = physics.inverse_operator_dense(problem.m0, block.frequency_hz)
    dense = physics.receiver_resolvent_matrix(
        problem.m0,
        block.frequency_hz,
        problem.geometry.receiver_indices,
        inverse_operator=inv_a,
    )
    sparse = physics.receiver_resolvent_matrix(
        problem.m0,
        block.frequency_hz,
        problem.geometry.receiver_indices,
    )
    assert np.allclose(sparse, dense, rtol=1e-10, atol=1e-11)


def test_direction_aware_reference_certificate_covers_realized_error():
    problem = build_synthetic_problem(shape=(8, 8), frequencies_hz=(3.0,), n_sources=1, n_receivers=6)
    physics = problem.physics
    block = problem.blocks[0]
    q = physics.source(block.source_index)
    surrogate = SmoothedHelmholtzSurrogate(physics, smoothing_sigma=0.7, model_scale=1.01)
    _, sg, _, _ = surrogate.objective_and_gradient(
        problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz
    )
    direction = -sg / np.linalg.norm(sg)
    receiver_matrix = physics.receiver_resolvent_matrix(
        problem.m0, block.frequency_hz, problem.geometry.receiver_indices
    )
    alpha = physics.receiver_resolvent_norm(
        problem.m0,
        block.frequency_hz,
        problem.geometry.receiver_indices,
        receiver_resolvent_matrix=receiver_matrix,
    )
    kappa = physics.directional_receiver_tangent_resolvent_norm(
        problem.m0,
        block.frequency_hz,
        problem.geometry.receiver_indices,
        direction,
        receiver_resolvent_matrix=receiver_matrix,
    )
    cert = certify_direction(
        physics=physics,
        surrogate=surrogate,
        m=problem.m0,
        q=q,
        observed=block.observed,
        receiver_indices=problem.geometry.receiver_indices,
        frequency_hz=block.frequency_hz,
        direction=direction,
        stability_mode="directional",
        receiver_resolvent_matrix=receiver_matrix,
        receiver_resolvent_norm=alpha,
        directional_tangent_resolvent_norm=kappa,
        validate_exact=True,
    )
    assert cert.realized_error is not None
    assert cert.directional_error_bound + 1e-9 >= cert.realized_error
