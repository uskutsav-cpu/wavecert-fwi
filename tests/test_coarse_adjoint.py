import json
from dataclasses import replace

import numpy as np
import pytest

from wavecert.certificates.coarse_adjoint import METHODS, PODCoarseAdjointCertifier
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.surrogates.pod import PODGalerkinSurrogate


def _case(*, full_basis=False, rank=20):
    p = Helmholtz2D(shape=(8, 8), damping_width=2, damping_strength=2.0)
    rng = np.random.default_rng(2741)
    basis = np.eye(p.n) if full_basis else np.linalg.qr(rng.normal(size=(p.n, 8)))[0]
    surrogate = PODGalerkinSurrogate(p, basis)
    engine = FastCoarseStability(p, rank)
    m = rng.uniform(0.20, 0.23, p.n)
    receivers = (3, 5, 5, 10)
    blocks = tuple(SurveyBlock(source, frequency,
        p.restrict(p.solve_state(m*1.02, p.source(source), frequency), receivers), str(i))
        for i, (source, frequency) in enumerate(((10, 3.0), (17, 4.0))))
    return p, surrogate, engine, m, receivers, blocks


def test_explicit_physical_pod_adapter_matches_unchanged_objective_gradient_and_counts():
    p, surrogate, engine, m, receivers, blocks = _case()
    expected = [surrogate.objective_and_gradient(m, p.source(b.source_index), b.observed,
        receivers, b.frequency_hz) for b in blocks]
    certifier = PODCoarseAdjointCertifier(p, surrogate, engine)
    result = certifier.certify_survey(m, blocks, receivers, method="uncorrected_oriented")
    assert result.approximate_objective == pytest.approx(sum(row[0] for row in expected), rel=1e-12)
    assert np.allclose(result.approximate_gradient, np.sum([row[1] for row in expected], axis=0), rtol=1e-12, atol=1e-13)
    for block, reference in zip(result.block_certificates, expected, strict=True):
        assert np.allclose(block.state, reference[2], rtol=1e-12, atol=1e-13)
        assert np.allclose(block.physical_gradient, reference[1], rtol=1e-12, atol=1e-13)
    assert result.costs["pod"]["reduced_forward_rhs"] == 2
    assert result.costs["pod"]["reduced_adjoint_rhs"] == 2
    assert result.costs["coarse_stability"]["online_dct_I_real_transforms"] == 2
    assert result.costs["reference_solver"]["factorizations"] == 0
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("method", METHODS)
def test_component_and_shared_direction_bounds_cover_independent_fine_adjoint_derivatives(method, monkeypatch):
    p, surrogate, engine, m, receivers, blocks = _case()
    exact_rows = [p.objective_and_gradient(m, p.source(b.source_index), b.observed,
        receivers, b.frequency_hz) for b in blocks]
    certifier = PODCoarseAdjointCertifier(p, surrogate, engine)

    def no_fine_solve(*args, **kwargs):
        raise AssertionError("candidate certificate must never call a fine solver")

    with monkeypatch.context() as context:
        context.setattr(Helmholtz2D, "_solve", no_fine_solve)
        result = certifier.certify_survey(m, blocks, receivers, method=method)
    assert result.available
    aggregate_exact = 0.0
    for block, fine, identity in zip(result.block_certificates, exact_rows, blocks, strict=True):
        exact_dd = float(np.dot(fine[1], result.direction))
        aggregate_exact += exact_dd
        assert abs(exact_dd-block.approximate_directional_derivative) <= block.directional_error_bound+1e-9
        stability = engine.evaluate(m, identity.frequency_hz)
        actual_u = stability.component_norms(fine[2]-block.state)
        residual = fine[3]
        exact_lambda = p._solve(m, p.inject_receivers(residual, receivers), identity.frequency_hz,
            category="adjoint_rhs", adjoint=True)
        actual_lambda = stability.component_norms(exact_lambda-block.adjoint)
        assert np.all(actual_u <= np.asarray(block.eta_u_components)+1e-9)
        assert np.all(actual_lambda <= np.asarray(block.eta_lambda_components)+1e-9)
        if method == "coarse_corrected_oriented":
            assert block.correction_applied and not block.correction_skipped
            assert block.primal_residual_components[0] < 1e-9
            assert block.adjoint_residual_components[0] < 1e-9
    assert abs(aggregate_exact-result.approximate_directional_derivative) <= result.directional_error_bound+1e-9
    assert result.exact_fallback_required == (not result.certified_descent)
    assert all(result.costs["coarse_stability"][name] == 0 for name in (
        "fine_grid_factorizations", "fine_grid_rhs", "fine_grid_dense_svd"))


def test_complete_basis_exact_limit_accepts_without_claiming_reduction_or_timing():
    p, surrogate, engine, m, receivers, blocks = _case(full_basis=True)
    result = PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(m, blocks, receivers)
    assert result.available and result.certified_descent
    assert not result.exact_fallback_required
    assert result.directional_error_bound < -result.approximate_directional_derivative
    assert result.costs["dimensions"]["pod"] == p.n  # An algebraic exact-limit test only.


def test_unavailable_block_retains_raw_pod_and_abstains_on_entire_aggregate():
    p, surrogate, _, m, receivers, blocks = _case()
    engine = FastCoarseStability(p, 1)
    shifted = SurveyBlock(blocks[1].source_index, 80.0, blocks[1].observed, "unavailable")
    raw = surrogate.objective_and_gradient(m, p.source(shifted.source_index), shifted.observed, receivers, 80.0)
    result = PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(m, (blocks[0], shifted), receivers)
    assert not result.available and not result.certified_descent and result.exact_fallback_required
    assert result.directional_error_bound is result.certified_upper_derivative is None
    failed = result.block_certificates[1]
    assert failed.correction_skipped and not failed.correction_applied
    assert np.allclose(failed.state, raw[2])
    assert np.allclose(failed.physical_gradient, raw[1])
    assert np.allclose(result.approximate_gradient, sum(b.physical_gradient for b in result.block_certificates))
    assert result.costs["pod"]["reduced_forward_rhs"] == 2
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_approximate_proposal_is_not_an_exact_stationarity_claim():
    p, surrogate, engine, m, receivers, blocks = _case()
    block = blocks[0]
    observed = surrogate.state(m, p.source(block.source_index), block.frequency_hz)[list(receivers)]
    result = PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(m,
        (SurveyBlock(block.source_index, block.frequency_hz, observed, "zero"),), receivers,
        method="uncorrected_oriented")
    assert result.available and not result.certified_descent and result.exact_fallback_required
    assert np.all(result.direction == 0)
    assert "no exact stationarity" in result.reason


def test_nonfinite_internal_physical_gradient_returns_strict_json_abstention(monkeypatch):
    p, surrogate, engine, m, receivers, blocks = _case()
    certifier = PODCoarseAdjointCertifier(p, surrogate, engine)
    monkeypatch.setattr(certifier, "_pod_fields", lambda *args: (
        np.full(p.n, 1e180, dtype=complex), np.full(p.n, 1e180, dtype=complex), 1.0))
    result = certifier.certify_survey(m, blocks, receivers, method="uncorrected_oriented")
    assert not result.available and not result.certified_descent and result.exact_fallback_required
    assert result.approximate_gradient is None and result.approximate_directional_derivative is None
    assert np.all(result.direction == 0)
    json.dumps(result.to_dict(), allow_nan=False)


def test_finite_block_bounds_with_overflowing_sum_abstain_with_strict_json(monkeypatch):
    p, surrogate, engine, m, receivers, blocks = _case()
    context_type = type(engine.evaluate(m, 3.0))
    monkeypatch.setattr(context_type, "component_norms", lambda *args: np.array([.02, 0.0]))
    monkeypatch.setattr(context_type, "oriented_error_bounds", lambda *args: np.zeros(2))
    monkeypatch.setattr(context_type, "oriented_adjoint_error_bounds", lambda *args: np.array([1e307, 0.0]))
    result = PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(
        m, blocks, receivers, method="uncorrected_oriented")
    assert all(block.available and np.isfinite(block.directional_error_bound) for block in result.block_certificates)
    assert not result.available and not result.certified_descent and result.exact_fallback_required
    assert result.directional_error_bound is result.certified_upper_derivative is None
    assert "nonfinite aggregate" in result.reason
    json.dumps(result.to_dict(), allow_nan=False)
    # An independently requested export of an overflowing block upper must
    # likewise return null instead of a JSON Infinity value.
    block = replace(result.block_certificates[0], approximate_directional_derivative=1e308,
                    directional_error_bound=1e308)
    assert block.certified_upper_derivative is None and not block.available
    json.dumps(block.to_dict(), allow_nan=False)


def test_opposing_finite_block_gradients_cannot_export_overflowing_block_derivatives(monkeypatch):
    p, surrogate, _, m, _, _ = _case()
    engine = FastCoarseStability(p, 1)
    certifier = PODCoarseAdjointCertifier(p, surrogate, engine)
    fields = iter(((np.full(p.n, 1e154), np.full(p.n, 1e154), 1.0),
                   (np.full(p.n, 1e154), np.full(p.n, -1e154), 1.0),
                   (np.ones(p.n), np.ones(p.n), 1.0)))
    monkeypatch.setattr(certifier, "_pod_fields", lambda *args: next(fields))
    blocks = tuple(SurveyBlock(10, frequency, np.zeros(1, dtype=complex), str(frequency))
                   for frequency in (80., 81., 82.))
    result = certifier.certify_survey(m, blocks, (3,), method="uncorrected_oriented")
    assert np.allclose(result.approximate_gradient, np.ones(p.n))
    assert result.approximate_directional_derivative is None  # No partial sum.
    assert not result.available and result.exact_fallback_required
    assert result.block_certificates[0].approximate_directional_derivative is None
    assert result.block_certificates[1].approximate_directional_derivative is None
    assert result.block_certificates[2].approximate_directional_derivative == pytest.approx(-8)
    json.dumps(result.to_dict(), allow_nan=False)


def test_model_and_actual_stability_contexts_are_bound_to_result_exports():
    p, surrogate, engine, m, receivers, blocks = _case()
    result = PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(m, blocks, receivers)
    exported = result.to_dict()
    assert set(exported["stability_contexts"]) == {"3.0", "4.0"}
    for context in exported["stability_contexts"].values():
        assert context["model_sha256"] == result.model_sha256
        assert context["operator_signature"] == engine.operator_signature
        assert context["available"]
    json.dumps(exported, allow_nan=False)


def test_interval_covers_independent_finite_difference_of_fine_objective():
    p, surrogate, engine, m, receivers, blocks = _case(full_basis=True)
    result = PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(m, blocks, receivers)
    assert result.certified_descent

    def objective(model):
        return sum(p.objective(model, p.source(b.source_index), b.observed, receivers, b.frequency_hz)
                   for b in blocks)

    step = 1e-5
    finite_difference = (objective(m+step*result.direction)-objective(m-step*result.direction))/(2*step)
    assert finite_difference < 0
    assert np.isclose(finite_difference, result.approximate_directional_derivative, rtol=1e-6, atol=1e-10)


def test_mismatched_reference_snapshot_is_rejected():
    p, surrogate, _, _, _, _ = _case()
    incompatible = FastCoarseStability(Helmholtz2D(shape=p.shape, spacing=0.06), 20)
    with pytest.raises(ValueError, match="snapshots"):
        PODCoarseAdjointCertifier(p, surrogate, incompatible)


@pytest.mark.parametrize("change", ["nan_model", "model_shape", "observed", "source", "receiver", "duplicate"])
def test_invalid_survey_data_fail_explicitly(change):
    p, surrogate, engine, m, receivers, blocks = _case()
    if change == "nan_model":
        m[0] = np.nan
    elif change == "model_shape":
        m = m.reshape(p.shape)
    elif change == "observed":
        blocks = (SurveyBlock(10, 3.0, np.zeros((4, 1)), "bad"),)
    elif change == "source":
        blocks = (SurveyBlock(True, 3.0, np.zeros(4), "bad"),)
    elif change == "receiver":
        receivers = (True, 5, 5, 10)
    else:
        blocks = (*blocks, blocks[0])
    with pytest.raises(ValueError):
        PODCoarseAdjointCertifier(p, surrogate, engine).certify_survey(m, blocks, receivers)
