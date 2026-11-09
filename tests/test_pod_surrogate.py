from __future__ import annotations

import numpy as np
import pytest

from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.pod import PODGalerkinSurrogate


def _case(rank=9, *, cache=3):
    rng = np.random.default_rng(128)
    physics = Helmholtz2D(shape=(6, 7), spacing=.08, damping_width=2, damping_strength=1.7)
    basis, _ = np.linalg.qr(rng.normal(size=(physics.n, rank)))
    surrogate = PODGalerkinSurrogate(physics, basis, factorization_cache_size=cache, mass_cache_size=cache)
    model = .18 + .04 * rng.random(physics.n)
    direction = rng.normal(size=physics.n)
    direction /= np.linalg.norm(direction)
    q = physics.source(8, amplitude=.8 + .35j) + physics.source(19, amplitude=-.4 + .2j)
    receivers = (3, 8, 8, 16, 22, 22)
    observations = rng.normal(size=len(receivers)) + 1j * rng.normal(size=len(receivers))
    return physics, surrogate, model, direction, q, receivers, observations


def test_reduced_state_jvp_and_complex_adjoint_match_independent_finite_differences():
    _, surrogate, model, direction, q, receivers, observations = _case()
    frequency = 2.2
    state, tangent = surrogate.state_and_jvp(model, direction, q, frequency)
    epsilon = 1e-5
    difference = (surrogate.state(model + epsilon * direction, q, frequency)
                  - surrogate.state(model - epsilon * direction, q, frequency)) / (2 * epsilon)
    np.testing.assert_allclose(tangent, difference, rtol=2e-7, atol=2e-9)
    objective, gradient, gradient_state, residual = surrogate.objective_and_gradient(model, q, observations, receivers, frequency)
    np.testing.assert_allclose(state, gradient_state)
    assert objective == pytest.approx(.5 * float(np.vdot(residual, residual).real))
    fd = (surrogate.objective(model + epsilon * direction, q, observations, receivers, frequency)
          - surrogate.objective(model - epsilon * direction, q, observations, receivers, frequency)) / (2 * epsilon)
    assert np.dot(gradient, direction) == pytest.approx(fd, rel=2e-7, abs=2e-9)


def test_data_vjp_is_real_adjoint_for_complex_sources_duals_and_repeated_receivers():
    _, surrogate, model, direction, q, receivers, _ = _case()
    dual = np.asarray([1 + 2j, -.3 + .7j, .8 - .2j, -.4 - 1.3j, 1.5 + .4j, -.6 + .9j])
    tangent = surrogate.jvp(model, direction, q, 3.4)
    gradient = surrogate.data_vjp(model, q, receivers, 3.4, dual)
    lhs = float(np.real(np.vdot(dual, tangent[list(receivers)])))
    assert np.dot(gradient, direction) == pytest.approx(lhs, rel=3e-12, abs=3e-12)


def test_identity_basis_recovers_fine_discrete_state_tangent_and_gradient():
    physics, _, model, direction, q, receivers, observations = _case()
    surrogate = PODGalerkinSurrogate(physics, np.eye(physics.n))
    frequency = 1.6
    state, tangent = surrogate.state_and_jvp(model, direction, q, frequency)
    exact = physics.solve_state(model, q, frequency)
    exact_tangent = physics.solve_tangent(model, exact, direction, frequency)
    np.testing.assert_allclose(state, exact, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, exact_tangent, rtol=2e-12, atol=2e-12)
    reduced_objective, reduced_gradient, _, _ = surrogate.objective_and_gradient(model, q, observations, receivers, frequency)
    exact_objective, exact_gradient, _, _ = physics.objective_and_gradient(model, q, observations, receivers, frequency)
    assert reduced_objective == pytest.approx(exact_objective, rel=2e-12)
    np.testing.assert_allclose(reduced_gradient, exact_gradient, rtol=3e-12, atol=3e-12)


def test_content_cache_reuses_conjugate_factor_and_frequency_independent_mass_then_invalidates_mutation():
    physics, surrogate, model, direction, q, receivers, observations = _case(cache=1)
    surrogate.state_and_jvp(model, direction, q, 2.)
    surrogate.objective_and_gradient(model.copy(), q, observations, receivers, 2.)
    costs = surrogate.cost_snapshot()
    assert costs["reduced_factorizations"] == costs["mass_assemblies"] == 1
    assert costs["reduced_forward_rhs"] == 2 and costs["reduced_tangent_rhs"] == costs["reduced_adjoint_rhs"] == 1
    surrogate.state(model, q, 2.5)
    costs = surrogate.cost_snapshot()
    assert costs["reduced_factorizations"] == 2 and costs["mass_assemblies"] == 1 and costs["mass_cache_hits"] == 1
    model *= 1.07
    changed = surrogate.state(model, q, 2.)
    fresh = PODGalerkinSurrogate(physics, surrogate.basis).state(model, q, 2.)
    np.testing.assert_allclose(changed, fresh)
    assert surrogate.cost_snapshot()["mass_assemblies"] == 2
    assert len(surrogate._factors) <= 1 and len(surrogate._masses) <= 1
    assert "forward_rhs" not in costs  # Reduced work is never mislabeled as fine-PDE RHS.


def test_zero_cache_still_shares_factor_within_one_combined_operation_and_counts_rhs_columns():
    _, surrogate, model, direction, q, _, _ = _case(cache=0)
    surrogate.state_and_jvp(model, direction, q, 2.)
    assert surrogate.cost_snapshot()["reduced_factorizations"] == 1
    _, _, factor = surrogate._prepare(model, 2.)
    surrogate._solve(factor, np.ones((surrogate.reduced_dimension, 3)), category="reduced_tangent_rhs")
    assert surrogate.cost_snapshot()["reduced_tangent_rhs"] == 4
    assert not surrogate._factors and not surrogate._masses


def test_basis_is_copied_read_only_and_archive_preserves_operator_and_prefix(tmp_path):
    physics, original, model, _, q, _, _ = _case()
    basis = original.basis.copy()
    surrogate = PODGalerkinSurrogate(physics, basis)
    reference = surrogate.state(model, q, 2.)
    basis[:] = 0
    np.testing.assert_allclose(surrogate.state(model, q, 2.), reference)
    with pytest.raises(ValueError):
        surrogate.basis[0, 0] = 0
    path = tmp_path / "basis.npz"
    surrogate.save(path, metadata={"scope": "unit test"})
    loaded = PODGalerkinSurrogate.load(path)
    np.testing.assert_allclose(loaded.state(model, q, 2.), reference)
    prefix = PODGalerkinSurrogate.load(path, rank=4)
    assert prefix.reduced_dimension == 4
    np.testing.assert_array_equal(prefix.basis, surrogate.basis[:, :4])


def test_external_physics_and_exported_configuration_cannot_corrupt_saved_operator_identity(tmp_path):
    physics, surrogate, model, _, q, _, _ = _case()
    reference = surrogate.state(model, q, 2.)
    original_config = surrogate.physics_config
    exported = surrogate.physics_config
    exported["spacing"] = .2
    exported["shape"][0] = 99
    physics.spacing = .3
    physics.damping_strength = 15
    assert surrogate.physics_config == original_config
    np.testing.assert_allclose(surrogate.state(model, q, 2.), reference)
    path = tmp_path / "preserved_operator.npz"
    surrogate.save(path)
    restored = PODGalerkinSurrogate.load(path)
    assert restored.physics_config == original_config
    np.testing.assert_allclose(restored.state(model, q, 2.), reference)


@pytest.mark.parametrize("setting", [True, -1, .5, np.nan])
def test_cache_limits_reject_noninteger_or_negative_values(setting):
    physics, surrogate, *_ = _case()
    with pytest.raises(ValueError, match="nonnegative integer"):
        PODGalerkinSurrogate(physics, surrogate.basis, factorization_cache_size=setting)


def test_invalid_basis_model_frequency_and_receiver_inputs_fail_explicitly():
    physics, surrogate, model, direction, q, _, _ = _case()
    with pytest.raises(ValueError, match="orthonormal"):
        PODGalerkinSurrogate(physics, 2 * surrogate.basis)
    with pytest.raises(ValueError, match="real two-dimensional"):
        PODGalerkinSurrogate(physics, surrogate.basis.astype(complex))
    for bad_model in (model * 0, np.full_like(model, np.nan), model.astype(complex)):
        with pytest.raises(ValueError):
            surrogate.state(bad_model, q, 2.)
    for frequency in (0, -1, np.nan, np.inf, 1e200):
        with pytest.raises(ValueError):
            surrogate.state(model, q, frequency)
    with pytest.raises(ValueError, match="real"):
        surrogate.jvp(model, direction.astype(complex), q, 2.)
    with pytest.raises(ValueError, match="receiver indices"):
        surrogate.objective(model, q, np.zeros(1), (1.5,), 2.)
    objective, gradient, _, residual = surrogate.objective_and_gradient(model, q, np.asarray([], dtype=complex), (), 2.)
    assert objective == 0 and not len(residual)
    np.testing.assert_array_equal(gradient, np.zeros(physics.n))
