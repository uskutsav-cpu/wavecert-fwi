import numpy as np
import pytest
import scipy.sparse.linalg as spla

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.repair.adaptive import adaptive_hybrid_direction, global_fallback_direction
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


@pytest.mark.parametrize("size", [-1, 0.5, float("nan"), True])
def test_invalid_cache_budget_cannot_disable_bounded_eviction(size):
    physics = Helmholtz2D(shape=(7, 7), factorization_cache_size=size)
    with pytest.raises(ValueError, match="non-negative integer"):
        physics.solve_state(np.full(physics.n, 0.25), physics.source(10), 3.0)


def test_forward_adjoint_and_stability_share_one_factorization():
    physics = Helmholtz2D(shape=(7, 7))
    m = np.full(physics.n, 0.25)
    q = physics.source(10)
    receivers = (10, 12, 12, 14)
    exact = spla.spsolve(physics.operator(m, 3.0), q)
    value, gradient, state, _ = physics.objective_and_gradient(
        m, q, np.zeros(len(receivers)), receivers, 3.0,
    )
    receiver_map = physics.receiver_resolvent_matrix(m, 3.0, receivers)
    physics.directional_receiver_tangent_resolvent_norm(
        m, 3.0, receivers, np.ones(physics.n), receiver_resolvent_matrix=receiver_map,
    )
    assert np.allclose(state, exact)
    assert np.isfinite(gradient).all()
    assert value > 0
    cost = physics.cost_snapshot()
    assert cost["factorizations"] == 1
    assert cost["forward_rhs"] == 1
    assert cost["adjoint_rhs"] == 1
    assert cost["stability_rhs"] == 2 * len(receivers)
    assert cost["svd_evaluations"] == 1
    assert cost["cache_hits"] == 3


def test_cache_follows_model_contents_and_evicts_old_factors():
    physics = Helmholtz2D(shape=(6, 6), factorization_cache_size=2)
    m = np.full(physics.n, 0.25)
    q = physics.source(8)
    initial = physics.solve_state(m, q, 3.0)
    assert np.allclose(physics.solve_state(m.copy(), q, 3.0), initial)
    assert physics.cost_snapshot()["factorizations"] == 1
    m[8] *= 1.05
    changed = physics.solve_state(m, q, 3.0)
    assert not np.allclose(changed, initial)
    assert np.allclose(changed, spla.spsolve(physics.operator(m, 3.0), q))
    physics.solve_state(m, q, 4.0)
    physics.solve_state(m, q, 5.0)
    assert len(physics._factorizations) == 2
    physics.solve_state(m, q, 3.0)
    assert physics.cost_snapshot()["factorizations"] == 5
    physics.spacing = 0.06
    changed_geometry = physics.solve_state(m, q, 3.0)
    assert np.allclose(changed_geometry, spla.spsolve(physics.operator(m, 3.0), q))
    assert physics.cost_snapshot()["factorizations"] == 6


def test_forward_only_objective_does_not_pay_for_an_adjoint():
    physics = Helmholtz2D(shape=(6, 6))
    m = np.full(physics.n, 0.25)
    q = physics.source(8)
    receivers = (8, 9, 10)
    value = physics.objective(m, q, np.zeros(3), receivers, 3.0)
    assert physics.cost_snapshot()["adjoint_rhs"] == 0
    reference, *_ = physics.objective_and_gradient(m, q, np.zeros(3), receivers, 3.0)
    assert value == pytest.approx(reference)
    physics.reset_cost_counters(clear_cache=True)
    assert all(v == 0 for v in physics.cost_snapshot().values())
    assert not physics._factorizations


@pytest.mark.parametrize("policy", [adaptive_hybrid_direction, global_fallback_direction])
def test_directional_resolvent_is_shared_across_sources(monkeypatch, policy):
    problem = build_synthetic_problem(
        shape=(8, 8), frequencies_hz=(3.0,), n_sources=3, n_receivers=6,
    )
    physics = problem.physics
    surrogate = SmoothedHelmholtzSurrogate(physics, smoothing_sigma=0.8)
    original = physics.directional_receiver_tangent_resolvent_norm
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(physics, "directional_receiver_tangent_resolvent_norm", count)
    kwargs = {"max_exact_blocks": 0} if policy is adaptive_hybrid_direction else {}
    result = policy(
        physics=physics, surrogate=surrogate, m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, stability_mode="directional", **kwargs,
    )
    assert result.certificate_rounds == 1
    assert len(calls) == 1


def test_beta_policy_does_not_construct_receiver_stability_map(monkeypatch):
    problem = build_synthetic_problem(
        shape=(6, 6), frequencies_hz=(3.0,), n_sources=2, n_receivers=4,
    )
    surrogate = SmoothedHelmholtzSurrogate(problem.physics, smoothing_sigma=0.8)

    def unexpected(*args, **kwargs):
        raise AssertionError("beta certificate does not need receiver solves")

    monkeypatch.setattr(problem.physics, "receiver_resolvent_matrix", unexpected)
    adaptive_hybrid_direction(
        physics=problem.physics, surrogate=surrogate, m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, stability_mode="beta", max_exact_blocks=0,
    )


@pytest.mark.parametrize("scale", [0.0, -0.1, float("nan"), float("inf")])
def test_unsupported_scales_fail_before_deployment(scale):
    problem = build_synthetic_problem(
        shape=(6, 6), frequencies_hz=(3.0,), n_sources=1, n_receivers=4,
    )
    surrogate = SmoothedHelmholtzSurrogate(problem.physics, smoothing_sigma=0.8)
    shared = dict(
        physics=problem.physics, surrogate=surrogate, m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices,
    )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        adaptive_hybrid_direction(**shared, bound_scales={problem.blocks[0].label: scale})
    with pytest.raises(ValueError, match="finite and strictly positive"):
        global_fallback_direction(**shared, bound_scale=scale)
