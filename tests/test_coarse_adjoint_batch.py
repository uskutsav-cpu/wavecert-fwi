import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
from test_pod_batch import _case

from wavecert.certificates.coarse_adjoint import METHODS, PODCoarseAdjointCertifier
from wavecert.certificates.coarse_adjoint_batch import BatchedPODCoarseAdjointCertifier
from wavecert.physics.coarse_batch import BatchedCoarseOperations, column_norms
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.surrogates.pod import PODGalerkinSurrogate


def _certifiers(p, basis, rank=30):
    scalar = PODCoarseAdjointCertifier(p, PODGalerkinSurrogate(p, basis), FastCoarseStability(p, rank))
    batch = BatchedPODCoarseAdjointCertifier(p, PODGalerkinSurrogate(p, basis), FastCoarseStability(p, rank))
    return scalar, batch


@pytest.mark.parametrize("method", METHODS)
def test_all_scalar_and_batched_fields_bounds_and_shared_direction_agree_on_complex_nonsquare_survey(method, monkeypatch):
    p, basis, model, receivers, blocks = _case()
    scalar, batch = _certifiers(p, basis)
    with monkeypatch.context() as guard:
        guard.setattr(Helmholtz2D, "_solve", lambda *a, **k: pytest.fail("certificate performed a fine solve"))
        expected = scalar.certify_survey(model, blocks, receivers, method=method)
        result = batch.certify_survey(model, blocks, receivers, method=method)
    assert result.available == expected.available
    assert result.certified_descent == expected.certified_descent
    assert result.exact_fallback_required == expected.exact_fallback_required
    assert result.model_sha256 == expected.model_sha256 and result.stability_contexts == expected.stability_contexts
    assert np.allclose(result.direction, expected.direction, rtol=3e-11, atol=1e-11)
    assert np.allclose(result.approximate_gradient, expected.approximate_gradient, rtol=3e-11, atol=1e-11)
    for name in ("approximate_objective", "approximate_directional_derivative", "directional_error_bound", "certified_upper_derivative"):
        assert getattr(result, name) == pytest.approx(getattr(expected, name), rel=3e-11, abs=1e-10)
    for new, old in zip(result.block_certificates, expected.block_certificates, strict=True):
        assert np.allclose(new.state, old.state, rtol=3e-11, atol=1e-11)
        assert np.allclose(new.adjoint, old.adjoint, rtol=3e-11, atol=1e-11)
        assert np.allclose(new.physical_gradient, old.physical_gradient, rtol=3e-11, atol=1e-11)
        for name in ("approximate_objective", "approximate_directional_derivative", "directional_error_bound",
                     "primal_residual_components", "adjoint_residual_components", "eta_u_components", "eta_lambda_components",
                     "eta_receiver_state", "bound_terms"):
            assert np.allclose(getattr(new, name), getattr(old, name), rtol=3e-11, atol=1e-10), name
    assert result.costs["pod"]["reduced_forward_rhs"] == result.costs["pod"]["reduced_adjoint_rhs"] == 4
    assert result.costs["batch_dispatches"]["pod_forward_solve_dispatches"] == 2
    assert result.costs["coarse_stability"]["coarse_inverse_rhs"] == 60
    assert result.costs["coarse_stability"]["coarse_solve_calls"] <= expected.costs["coarse_stability"]["coarse_solve_calls"]
    if method != "uncorrected_generic_M":
        assert result.costs["coarse_stability"]["coarse_solve_calls"] < expected.costs["coarse_stability"]["coarse_solve_calls"]
    assert result.costs["coarse_stability"]["dst_I_real_transforms"] <= expected.costs["coarse_stability"]["dst_I_real_transforms"]
    assert all(v == 0 for v in result.costs["reference_solver"].values())
    json.dumps(result.to_dict(), allow_nan=False)


def test_batched_fft_source_axis_stencils_and_stable_column_norms_match_independent_vectors():
    p, _, model, _, _ = _case()
    context = FastCoarseStability(p, 30).evaluate(model, 3.)
    rng = np.random.default_rng(95)
    values = rng.normal(size=(p.n, 5))+1j*rng.normal(size=(p.n, 5))
    values[:, 2] = 0
    dispatches = {}
    operations = BatchedCoarseOperations(context, dispatches)
    projected = operations.project(values)
    assert np.allclose(projected, np.column_stack([context.project(v) for v in values.T]), rtol=1e-13, atol=1e-13)
    assert np.allclose(operations.lift(projected), np.column_stack([context.lift(v) for v in projected.T]), rtol=1e-13, atol=1e-13)
    for adjoint in (False, True):
        assert np.allclose(operations.apply_operator(values, adjoint=adjoint),
            np.column_stack([context.apply_operator(v, adjoint=adjoint) for v in values.T]), rtol=1e-13, atol=1e-11)
    assert dispatches["dst_I_dispatches"] == dispatches["idst_I_dispatches"] == 1
    assert np.allclose(column_norms(values*1e180), [la.norm(v*1e180) for v in values.T], rtol=2e-15)


def test_batched_bound_covers_independent_fine_adjoint_and_objective_finite_difference():
    p, basis, model, receivers, blocks = _case()
    _, batch = _certifiers(p, basis)
    result = batch.certify_survey(model, blocks, receivers)
    exact = [p.objective_and_gradient(model, p.source(b.source_index), b.observed, receivers, b.frequency_hz) for b in blocks]
    derivative = sum(float(row[1] @ result.direction) for row in exact)
    assert abs(derivative-result.approximate_directional_derivative) <= result.directional_error_bound+1e-9

    def objective(values):
        return sum(p.objective(values, p.source(b.source_index), b.observed, receivers, b.frequency_hz) for b in blocks)

    step = 1e-5
    finite_difference = (objective(model+step*result.direction)-objective(model-step*result.direction))/(2*step)
    assert finite_difference == pytest.approx(derivative, rel=1e-6, abs=1e-9)


def test_exact_full_basis_limit_has_same_positive_acceptance_away_from_gate_threshold():
    p, _, model, receivers, blocks = _case()
    scalar, batch = _certifiers(p, np.eye(p.n))
    old = scalar.certify_survey(model, blocks, receivers)
    new = batch.certify_survey(model, blocks, receivers)
    assert old.certified_descent and new.certified_descent
    assert old.certified_upper_derivative < -1e-6
    assert new.certified_upper_derivative == pytest.approx(old.certified_upper_derivative, rel=3e-11, abs=1e-9)
    assert np.allclose(new.direction, old.direction, rtol=3e-11, atol=1e-11)


def test_mixed_unavailable_frequency_retains_all_raw_blocks_and_scalar_fallback():
    p, basis, model, receivers, blocks = _case()
    blocks = blocks[:2]+(SurveyBlock(9, 80., blocks[2].observed, "unavailable"),)
    scalar, batch = _certifiers(p, basis)
    expected = scalar.certify_survey(model, blocks, receivers)
    result = batch.certify_survey(model, blocks, receivers)
    assert not result.available and not result.certified_descent and result.exact_fallback_required
    assert result.directional_error_bound is result.certified_upper_derivative is None
    assert len(result.block_certificates) == len(blocks)
    assert result.block_certificates[-1].correction_skipped
    assert np.allclose(result.approximate_gradient, expected.approximate_gradient, rtol=3e-11, atol=1e-10)
    assert np.allclose(result.block_certificates[-1].state, expected.block_certificates[-1].state, rtol=3e-11, atol=1e-10)
    json.dumps(result.to_dict(), allow_nan=False)


def test_batch_certifier_content_cache_reuse_and_new_model_work_are_actual():
    p, basis, model, receivers, blocks = _case()
    _, batch = _certifiers(p, basis)
    first = batch.certify_survey(model, blocks, receivers)
    again = batch.certify_survey(model.copy(), blocks, receivers)
    assert again.costs["coarse_stability"]["online_dct_I_real_transforms"] == 0
    assert again.costs["coarse_stability"]["coarse_lu_factorizations"] == 2
    assert again.costs["coarse_stability"]["coarse_inverse_rhs"] == 60
    assert np.array_equal(first.approximate_gradient, again.approximate_gradient)
    changed = model.copy()
    changed[0] *= 1.01
    fresh = batch.certify_survey(changed, blocks, receivers)
    assert fresh.costs["coarse_stability"]["online_dct_I_real_transforms"] == 2
    assert fresh.costs["pod"]["mass_assemblies"] == 1
    assert fresh.model_sha256 != first.model_sha256


def test_zero_uncorrected_proposal_abstains_without_stationarity_claim():
    p, basis, model, receivers, blocks = _case()
    surrogate = PODGalerkinSurrogate(p, basis)
    b = blocks[0]
    observed = surrogate.state(model, p.source(b.source_index), b.frequency_hz)[list(receivers)]
    new = SurveyBlock(b.source_index, b.frequency_hz, observed, "zero")
    result = BatchedPODCoarseAdjointCertifier(p, surrogate, FastCoarseStability(p, 30)).certify_survey(
        model, (new,), receivers, method="uncorrected_oriented")
    assert not result.certified_descent and result.exact_fallback_required
    assert np.all(result.direction == 0)
    assert "no exact stationarity" in result.reason
    json.dumps(result.to_dict(), allow_nan=False)


def test_nonfinite_physical_contraction_abstains_with_strict_json(monkeypatch):
    import wavecert.certificates.coarse_adjoint_batch as module

    p, basis, model, receivers, blocks = _case()
    _, batch = _certifiers(p, basis)
    fields = np.full((p.n, len(blocks)), 1e180, dtype=complex)
    groups = {f: tuple(i for i, b in enumerate(blocks) if b.frequency_hz == f) for f in (2., 3.)}
    monkeypatch.setattr(module, "_batched_pod_survey_fields", lambda *a, **k: SimpleNamespace(
        block_states=fields, block_adjoints=fields, group_indices=groups, batch_dispatches={}))
    result = batch.certify_survey(model, blocks, receivers, method="uncorrected_oriented")
    assert not result.available and result.exact_fallback_required
    assert result.approximate_gradient is result.approximate_directional_derivative is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_finite_opposing_gradients_with_nonfinite_block_derivatives_serialize_no_partial_sum(monkeypatch):
    import wavecert.certificates.coarse_adjoint_batch as module

    p, basis, model, _, _ = _case()
    _, batch = _certifiers(p, basis)
    frequency, omega2 = 80., (2*np.pi*80)**2
    blocks = tuple(SurveyBlock(s, frequency, np.zeros(1, dtype=complex), str(s)) for s in (3, 9, 17))
    large = np.sqrt(1e308/omega2)
    state = np.column_stack((np.full(p.n, large), np.full(p.n, large), np.ones(p.n)))
    adjoint = np.column_stack((np.full(p.n, large), np.full(p.n, -large), np.full(p.n, 1/omega2)))
    monkeypatch.setattr(module, "_batched_pod_survey_fields", lambda *a, **k: SimpleNamespace(
        block_states=state, block_adjoints=adjoint, group_indices={frequency: (0, 1, 2)}, batch_dispatches={}))
    result = batch.certify_survey(model, blocks, (5,), method="uncorrected_oriented")
    assert np.allclose(result.approximate_gradient, np.ones(p.n))
    assert result.approximate_directional_derivative is None
    assert result.block_certificates[0].approximate_directional_derivative is None
    assert result.block_certificates[1].approximate_directional_derivative is None
    assert not result.available and result.exact_fallback_required
    json.dumps(result.to_dict(), allow_nan=False)


def test_finite_block_bounds_that_overflow_in_aggregate_force_json_safe_fallback(monkeypatch):
    p, basis, model, receivers, blocks = _case()
    _, batch = _certifiers(p, basis)
    monkeypatch.setattr(BatchedCoarseOperations, "oriented_error_bounds", lambda self, rb, cp: np.zeros((2, rb.shape[1])))
    monkeypatch.setattr(BatchedCoarseOperations, "oriented_adjoint_error_bounds", lambda self, rb, *a:
        np.vstack((np.full(rb.shape[1], 1e307), np.zeros(rb.shape[1]))))
    monkeypatch.setattr(BatchedCoarseOperations, "component_norms", lambda self, values:
        np.vstack((np.full(values.shape[1], .02), np.zeros(values.shape[1]))))
    result = batch.certify_survey(model, blocks, receivers, method="uncorrected_oriented")
    assert all(block.available and np.isfinite(block.directional_error_bound) for block in result.block_certificates)
    assert not result.available and result.directional_error_bound is result.certified_upper_derivative is None
    assert result.exact_fallback_required
    json.dumps(result.to_dict(), allow_nan=False)
