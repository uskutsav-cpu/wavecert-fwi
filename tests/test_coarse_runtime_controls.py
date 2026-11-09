import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wavecert.experiments.coarse_runtime_controls import (
    METHODS,
    _measure_control,
    _method_order,
    _paired,
    _parity,
    load_runtime_inputs,
    run_runtime_controls,
    validate_runtime_protocol,
)
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.surrogates.pod import PODGalerkinSurrogate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/coarse_runtime_controls.json"


def test_tiny_controls_have_correct_public_api_parity_and_executed_work_without_real_timing(monkeypatch):
    physics = Helmholtz2D(shape=(8, 8), spacing=.05, damping_width=2, damping_strength=2.)
    config = dict(shape=list(physics.shape), spacing=physics.spacing,
                  damping_width=physics.damping_width, damping_strength=physics.damping_strength)
    rng = np.random.default_rng(1328)
    model, receivers = rng.uniform(.20, .23, physics.n), (3, 5, 5, 10)
    blocks = tuple(SurveyBlock(source, frequency,
        physics.restrict(physics.solve_state(model*1.02, physics.source(source), frequency), receivers), str(source))
        for source, frequency in ((10, 3.), (17, 4.)))
    references = [physics.objective_and_gradient(model, physics.source(b.source_index), b.observed,
        receivers, b.frequency_hz) for b in blocks]
    exact_gradient = sum(row[1] for row in references)
    outcomes = {}
    for method in METHODS:
        ticks = iter((10., 11., 13.))  # Stub clock: no measured timing evidence.
        if method != METHODS[0]:
            with monkeypatch.context() as guard:
                guard.setattr(Helmholtz2D, "_solve", lambda *a, **k: pytest.fail("POD control performed a fine solve"))
                result = _measure_control(method, physics_config=config, basis=np.eye(physics.n),
                    model=model, blocks=blocks, receivers=receivers, coarse_rank=20, clock=lambda ticks=ticks: next(ticks))
        else:
            result = _measure_control(method, physics_config=config, basis=np.eye(physics.n),
                model=model, blocks=blocks, receivers=receivers, coarse_rank=20, clock=lambda ticks=ticks: next(ticks))
        outcomes[method] = result
        assert result["setup_seconds"] == 1 and result["query_seconds"] == 2 and result["total_cold_seconds"] == 3
        assert result["objective"] == pytest.approx(sum(row[0] for row in references), rel=1e-9, abs=1e-12)
        assert np.allclose(result["gradient"], exact_gradient, rtol=1e-9, atol=1e-12)
    exact_cost = outcomes[METHODS[0]]["query_costs"]["reference_solver"]
    assert outcomes[METHODS[0]]["exact_expected_rhs"] == 4
    assert exact_cost["forward_rhs"] == exact_cost["adjoint_rhs"] == exact_cost["factorizations"] == 2
    assert exact_cost["svd_evaluations"] == 0
    raw = outcomes[METHODS[1]]
    assert raw["setup_costs"]["pod_operator_projection"]["stiffness_projections"] == 1
    assert raw["setup_costs"]["pod_operator_projection"]["damping_projections"] == 1
    assert raw["setup_costs"]["pod_basis_orthogonality_gram_evaluations"] == 1
    assert raw["query_costs"]["pod"]["mass_assemblies"] == 1
    assert raw["query_costs"]["pod"]["reduced_forward_rhs"] == raw["query_costs"]["pod"]["reduced_adjoint_rhs"] == 2
    corrected = outcomes[METHODS[2]]
    assert corrected["certificate"].certified_descent
    coarse = corrected["query_costs"]["coarse_stability"]
    setup = corrected["setup_costs"]["coarse_stability"]
    assert setup["offline_dct_I_real_transforms"] == 2
    assert setup["coarse_multiplication_matrices"] == 2
    assert coarse["online_dct_I_real_transforms"] == 2
    assert coarse["coarse_lu_factorizations"] == 2
    assert coarse["coarse_inverse_rhs"] == 40
    assert coarse["coarse_receiver_rhs"] == 8  # Four receiver slots, including the repeated row, at two frequencies.
    assert coarse["coarse_correction_rhs"] == coarse["coarse_oriented_rhs"] == 4
    assert coarse["coarse_solve_rhs"] == 56
    assert coarse["fine_grid_factorizations"] == coarse["fine_grid_rhs"] == coarse["fine_grid_dense_svd"] == 0
    assert all(value == 0 for value in corrected["query_costs"]["reference_solver"].values())
    # Each new invocation pays cold construction/model work again.
    ticks = iter((0., 1., 3.))
    repeated = _measure_control(METHODS[2], physics_config=config, basis=np.eye(physics.n),
        model=model, blocks=blocks, receivers=receivers, coarse_rank=20, clock=lambda: next(ticks))
    assert repeated["query_costs"] == corrected["query_costs"]


def test_runtime_input_check_has_no_predictions_or_timing(monkeypatch):
    import wavecert.experiments.coarse_runtime_controls as runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("input audit must not predict, solve, construct models or time controls")

    monkeypatch.setattr(runtime, "_measure_control", forbidden)
    monkeypatch.setattr(runtime, "perf_counter", forbidden)
    monkeypatch.setattr(PODGalerkinSurrogate, "__init__", forbidden)
    monkeypatch.setattr(Helmholtz2D, "_solve", forbidden)
    arrays, basis, audit = load_runtime_inputs(json.loads(PROTOCOL.read_bytes()), ROOT)
    assert arrays["models"].shape == (8, 24, 24) and basis.shape == (576, 64)
    assert audit["calibration_or_test_arrays_loaded"] is False


@pytest.mark.parametrize("key,value", [("repetitions", 1), ("spectral_coarse_rank", 64),
                                       ("methods", [METHODS[2]]), ("heldout_arrays_loaded", True)])
def test_runtime_protocol_cannot_drop_repetitions_change_rank_or_cohort(key, value):
    protocol = json.loads(PROTOCOL.read_bytes())
    protocol[key] = value
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_runtime_protocol(protocol)


def test_fixed_validation_protocol_hash_mismatch_fails_closed():
    protocol = json.loads(PROTOCOL.read_bytes())
    protocol["validation_protocol"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="validation protocol SHA-256"):
        load_runtime_inputs(protocol, ROOT)


def test_method_order_is_balanced_for_each_case_and_pairs_keep_all_methods():
    for case in range(8):
        for method in METHODS:
            assert sorted(_method_order(rep, case).index(method) for rep in range(3)) == [0, 1, 2]
    rows = [{"repetition": 0, "case_index": 0, "method": method, "parent_id": "parent", "slice_id": "slice",
             "setup_seconds": 1., "query_seconds": 2., "total_cold_seconds": 3.} for method in METHODS]
    assert len(_paired(rows)) == 1
    with pytest.raises(ValueError, match="incomplete"):
        _paired(rows[:-1])
    assert _parity(None, None, None, None, 1e-9, 1e-10)
    assert not _parity(1., np.ones(2), 2., np.ones(2), 1e-9, 1e-10)


def test_previous_runtime_evidence_is_never_overwritten(tmp_path):
    output = tmp_path / "prior"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("preserve")
    with pytest.raises(ValueError, match="new or empty"):
        run_runtime_controls(PROTOCOL, output, root=ROOT)
    assert summary.read_text() == "preserve"


def test_numeric_abstention_cannot_mark_complete_even_when_repeat_parities_and_cohort_pass(tmp_path, monkeypatch):
    """Exercise complete runner bookkeeping with mocked outputs and a stub clock.

    No real method prediction or wall timing is performed. This specifically
    catches a None/None repetition parity masking a failed corrected proposal.
    """
    import wavecert.experiments.coarse_runtime_controls as runtime

    for key in runtime._THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    ticks = iter(float(i) for i in range(20))
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(ticks))
    mock_block = SimpleNamespace(approximate_directional_derivative=None, available=False,
        physical_gradient=None, to_dict=lambda: {"available": False})
    mock_certificate = SimpleNamespace(direction=np.zeros(576), approximate_directional_derivative=None,
        directional_error_bound=None, certified_upper_derivative=None, available=False,
        certified_descent=False, exact_fallback_required=True, block_certificates=(mock_block,)*24,
        stability_contexts={}, model_sha256="0"*64)

    def control(method, **kwargs):
        failed = method == METHODS[2]
        costs = {"reference_solver": {}, "coarse_stability": dict.fromkeys(
            ("fine_grid_factorizations", "fine_grid_rhs", "fine_grid_dense_svd"), 0)}
        return {"method": method, "objective": None if failed else 1.,
            "gradient": None if failed else np.ones(576), "certificate": mock_certificate if failed else None,
            "setup_seconds": 1., "query_seconds": 2., "total_cold_seconds": 3.,
            "setup_costs": {}, "query_costs": costs, "cumulative_costs": costs,
            "exact_frequency_strategies": {}, "exact_expected_rhs": None}

    def references(cases, *args):
        return [{"objective": 1., "gradient": np.ones(576),
                 "blocks": [{"gradient": np.ones(576)/24}]*24,
                 "stored_teacher_agreement": True} for case in cases], {}

    monkeypatch.setattr(runtime, "_measure_control", control)
    monkeypatch.setattr(runtime, "_offline_references", references)
    result = run_runtime_controls(PROTOCOL, tmp_path / "mock-results", root=ROOT)
    assert result["status"] == "checks-failed"
    assert result["checks"]["complete_fixed_cohort"] and result["checks"]["all_cold_repeat_parities"]
    assert not result["checks"]["finite_objectives_and_gradients"]
    assert all(value for key, value in result["checks"].items() if key != "finite_objectives_and_gradients")
