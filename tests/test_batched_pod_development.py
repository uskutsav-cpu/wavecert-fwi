"""Tiny synthetic parity and mocked bookkeeping; no project timing evidence."""

import copy
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wavecert.experiments import batched_pod_development as development
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/batched_pod_development_protocol.json"


def test_input_audit_does_not_construct_predict_or_time(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input audit must not construct/predict/time")
    monkeypatch.setattr(development, "PODGalerkinSurrogate", forbidden)
    monkeypatch.setattr(Helmholtz2D, "_solve", forbidden)
    monkeypatch.setattr(development, "execute_control", forbidden)
    monkeypatch.setattr(development, "perf_counter", forbidden)
    protocol, _, arrays, basis, audit = development.load_development_inputs(PROTOCOL, ROOT)
    assert protocol["methods"] == list(development.METHODS)
    assert arrays["models"].shape == (8, 24, 24) and basis.shape == (576, 64)
    assert audit["calibration_or_test_arrays_loaded"] is False
    assert audit["external_test_arrays_loaded"] is False


def test_protocol_and_production_bytes_are_immutable(tmp_path, monkeypatch):
    path = tmp_path / "changed.json"
    protocol = json.loads(PROTOCOL.read_bytes())
    protocol["timing"]["repetitions"] = 1
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol bytes changed"):
        development.load_development_inputs(path, ROOT)
    monkeypatch.setattr(development, "sha256_file", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="production source"):
        development.load_development_inputs(PROTOCOL, ROOT)


def test_each_method_occupies_every_position_and_incomplete_pairs_fail():
    rows = []
    for repetition in range(6):
        for index in range(8):
            for method in development.method_order(repetition, index):
                rows.append(dict(repetition=repetition, case_index=index, parent_id=str(index // 2),
                    slice_id=str(index), method=method, setup_seconds=1., query_seconds=2., total_cold_seconds=3.))
    for index in range(8):
        for method in development.METHODS:
            assert sorted(development.method_order(rep, index).index(method) for rep in range(6)) == list(range(6))
    assert len(development.paired_rows(rows)) == 48
    with pytest.raises(ValueError, match="incomplete"):
        development.paired_rows(rows[:-1])


@pytest.mark.parametrize("name", ["wavecert.experiments.batched_pod_development",
    "wavecert.experiments.coarse_runtime_controls", "wavecert.physics.coarse_batch"])
def test_foreign_cached_execution_modules_are_rejected(name, monkeypatch):
    module = import_module(name)
    monkeypatch.setattr(module, "__file__", "/foreign/checkout/module.py")
    with pytest.raises(ValueError, match="another checkout"):
        development.load_development_inputs(PROTOCOL, ROOT)


def _eligibility_inputs():
    rows, parity = [], []
    for index in range(8):
        for method in ("raw_POD_batch", *development.CANDIDATES):
            parity.append(dict(passes=True, batch_quality=dict(method=method, certified_descent=True,
                directional_bound_covers=True, all_block_bounds_cover=True)))
        for repetition in range(6):
            for method in development.METHODS:
                rows.append(dict(case_index=index, repetition=repetition, parent_id=str(index // 2), method=method,
                    finite_proposal=True, repeat_parity=True, certified_descent=True, certificate_available=True,
                    directional_bound_covers=True, all_block_bounds_cover=True,
                    total_cold_seconds=1. if method == "corrected_batch64" else 2.))
    return rows, parity


def test_cheaper_rank_is_vetoed_by_even_one_failed_slice_or_block():
    rows, parity = _eligibility_inputs()
    assert development.candidate_recommendation(rows, parity)["recommended_development_candidate"] == "corrected_batch64"
    changed = copy.deepcopy(rows)
    next(r for r in changed if r["method"] == "corrected_batch64")["all_block_bounds_cover"] = False
    result = development.candidate_recommendation(changed, parity)
    assert result["recommended_development_candidate"] == "corrected_batch96"
    assert not result["candidates"]["corrected_batch64"]["eligible"]
    changed_parity = copy.deepcopy(parity)
    next(r for r in changed_parity if r["batch_quality"]["method"] == "corrected_batch96")["passes"] = False
    result = development.candidate_recommendation(changed, changed_parity)
    assert result["recommended_development_candidate"] is None and "retain existing" in result["action"]
    assert len(changed) == 288  # Failures remain, rather than entering a smaller favorable cohort.


def test_cost_rule_uses_per_slice_medians_and_ties_prefer96():
    rows, parity = _eligibility_inputs()
    for row in rows:
        if row["method"] in development.CANDIDATES:
            row["total_cold_seconds"] = 10_000. if row["repetition"] == 0 else float(row["case_index"] + 1)
    result = development.candidate_recommendation(rows, parity)
    assert result["recommended_development_candidate"] == "corrected_batch96"
    for candidate in development.CANDIDATES:
        assert result["candidates"][candidate]["equal_parent_mean_slice_median_total_cold_seconds"] == 4.5


def test_tiny_scalar_batch_parity_and_costs_with_stub_clock_only():
    physics = Helmholtz2D(shape=(11, 10), spacing=.05, damping_width=2, damping_strength=2.)
    config = dict(shape=list(physics.shape), spacing=physics.spacing, damping_width=2, damping_strength=2.)
    rng = np.random.default_rng(348)
    basis = np.linalg.qr(rng.normal(size=(physics.n, 8)))[0]
    model, receivers = rng.uniform(.2, .23, physics.n), (3, 5, 5, 10)
    pairs = ((10, 3.), (17, 4.), (17, 3.), (10, 4.))  # Interleaved frequency groups.
    blocks = tuple(SurveyBlock(source, frequency,
        physics.restrict(physics.solve_state(model * 1.02, physics.source(source), frequency), receivers), str(i))
        for i, (source, frequency) in enumerate(pairs))
    case = dict(model=model, blocks=blocks, receivers=receivers)
    for scalar, batch in (("raw_POD_scalar", "raw_POD_batch"),
                          ("corrected_scalar96", "corrected_batch96"),
                          ("corrected_scalar64", "corrected_batch64")):
        outcomes = []
        for method in (scalar, batch):
            ticks = iter((0., 1., 3.))
            result = development.execute_control(method, config=config, basis=basis, case=case,
                                                   clock=lambda ticks=ticks: next(ticks))
            assert result["setup_seconds"] == 1 and result["query_seconds"] == 2 and result["total_cold_seconds"] == 3
            if method == "raw_POD_scalar":
                extra = development._add_paid_scalar_adjoint_fields(result, case)
                assert extra["pod"]["reduced_adjoint_rhs"] == 4
            outcomes.append(result)
        comparison = development.compare_results(*outcomes, rtol=1e-9, atol=1e-10)
        assert comparison["passes"], comparison
        for result in outcomes:
            assert result["query_costs"]["pod"]["reduced_forward_rhs"] == 4
            assert result["query_costs"]["pod"]["reduced_adjoint_rhs"] == 4
            assert all(value == 0 for value in result["query_costs"]["reference_solver"].values())
        if batch.startswith("corrected_"):
            rank = 64 if batch.endswith("64") else 96
            assert outcomes[1]["query_costs"]["coarse_stability"]["coarse_inverse_rhs"] == 2 * rank
            assert outcomes[1]["setup_costs"]["coarse_stability"]["offline_dct_I_real_transforms"] == 2


def test_failed_preflight_starts_no_comparative_timing(tmp_path, monkeypatch):
    for key in development.THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(development, "_offline_references", lambda *args: ([{"stored_teacher_agreement": True}] * 8, {}))
    def parity_failure(*args, **kwargs):
        raise ValueError("synthetic parity failed")
    def forbidden(*args, **kwargs):
        raise AssertionError("comparative timing must not start after a parity failure")
    monkeypatch.setattr(development, "preflight", parity_failure)
    monkeypatch.setattr(development, "execute_control", forbidden)
    monkeypatch.setattr(development, "perf_counter", forbidden)
    output = tmp_path / "failed-preflight"
    with pytest.raises(ValueError, match="synthetic parity failed"):
        development.run_development(PROTOCOL, output, root=ROOT)
    status = json.loads((output / "status.json").read_bytes())
    assert status["status"] == "failed" and status["completed_cold_calls"] == 0


def test_paired_missing_proposals_fail_real_preflight_before_any_timed_call(tmp_path, monkeypatch):
    """None/None numeric comparison is insufficient even when it agrees."""
    for key in development.THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    cases = [dict(case_index=i, parent_id=str(i // 2), slice_id=str(i), model=np.ones(2),
                  blocks=(None,) * 24, receivers=(0,)) for i in range(8)]
    reference = dict(objective=1., gradient=np.ones(2),
                     blocks=[dict(objective=1 / 24, gradient=np.ones(2) / 24)] * 24,
                     stored_teacher_agreement=True)
    monkeypatch.setattr(development, "_cases", lambda arrays: cases)
    monkeypatch.setattr(development, "_offline_references", lambda *args: ([reference] * 8, {}))
    monkeypatch.setattr(development, "_add_paid_scalar_adjoint_fields", lambda *args: {})
    monkeypatch.setattr(development, "perf_counter", lambda: pytest.fail("no comparative timing before finite parity"))
    untimed = []

    def missing(method, *, clock=None, **kwargs):
        assert clock is None, "missing paired outputs reached a timed call"
        untimed.append(method)
        return dict(method=method, objective=None, gradient=None, certificate=None,
            states=np.zeros((2, 24)), adjoints=np.zeros((2, 24)),
            block_gradients=None, block_objectives=None, block_keys=[(i, float(i)) for i in range(24)],
            setup_costs={}, query_costs={}, batch_dispatches={})
    monkeypatch.setattr(development, "execute_control", missing)
    output = tmp_path / "paired-missing"
    with pytest.raises(ValueError, match="finite complete fields"):
        development.run_development(PROTOCOL, output, root=ROOT)
    status = json.loads((output / "status.json").read_bytes())
    assert status["status"] == "failed" and status["completed_cold_calls"] == 0
    records = json.loads((output / "parity_checks.json").read_bytes())
    assert len(records) == 24 and len(untimed) == 48
    assert all(not record["passes"] and not record["finite_fields"]["scalar"]["passes"]
               and not record["finite_fields"]["batch"]["passes"] for record in records)


def test_existing_output_cannot_be_overwritten(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    retained = output / "keep.txt"
    retained.write_text("evidence")
    with pytest.raises(ValueError, match="new directory"):
        development.run_development(PROTOCOL, output, root=ROOT)
    assert retained.read_text() == "evidence"


@pytest.mark.parametrize("failure_mode", [None, "proposal", "proof"])
def test_mocked_whole_study_retains_all_rows_and_never_selects_numeric_failure(tmp_path, monkeypatch, failure_mode):
    """All model outputs/times below are mocks, not project measurements."""
    for key in development.THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    cases = [dict(case_index=i, parent_id=str(i // 2), slice_id=str(i), model=np.ones(2),
                  blocks=(), receivers=()) for i in range(8)]
    reference = dict(objective=1., gradient=np.ones(2),
                     blocks=[dict(objective=1 / 24, gradient=np.ones(2) / 24)] * 24,
                     stored_teacher_agreement=True)
    monkeypatch.setattr(development, "_cases", lambda arrays: cases)
    monkeypatch.setattr(development, "_offline_references", lambda *args: ([reference] * 8, {}))
    _, parity = _eligibility_inputs()
    monkeypatch.setattr(development, "preflight", lambda *args, **kwargs: (parity, []))
    monkeypatch.setattr(development, "perf_counter", lambda: pytest.fail("stub controls must not measure wall time"))

    def control(method, **kwargs):
        proposal_failed = failure_mode == "proposal" and method == "corrected_batch64"
        failed = failure_mode is not None and method == "corrected_batch64"
        gradient = None if proposal_failed else np.ones(2)
        direction = np.zeros(2) if proposal_failed else -np.ones(2) / np.sqrt(2)
        corrected = method.startswith("corrected_")
        block_keys = [(i, float(i)) for i in range(24)]
        blocks = []
        for source, frequency in block_keys:
            blocks.append(SimpleNamespace(source_index=source, frequency_hz=frequency,
                physical_gradient=None if proposal_failed else np.ones(2) / 24,
                directional_error_bound=None if failed else .001, available=not failed,
                reason="nonfinite mock" if failed else None, to_dict=lambda: {}))
        certificate = None if not corrected else SimpleNamespace(
            direction=direction, approximate_directional_derivative=None if proposal_failed else -np.sqrt(2),
            directional_error_bound=None if failed else .024, available=not failed,
            certified_descent=not failed, exact_fallback_required=failed,
            certified_upper_derivative=None if failed else -np.sqrt(2) + .024,
            block_certificates=blocks, stability_contexts={}, model_sha256="0" * 64)
        if certificate is not None:
            certificate.reason = "nonfinite mock" if failed else None
        fine = dict(factorizations=0, forward_rhs=0, adjoint_rhs=0)
        costs = dict(reference_solver=fine, coarse_stability=dict.fromkeys(
            ("fine_grid_factorizations", "fine_grid_rhs", "fine_grid_dense_svd"), 0))
        total = 1. if method == "corrected_batch64" else 2.
        return dict(method=method, objective=None if proposal_failed else 1., gradient=gradient, certificate=certificate,
            block_gradients=None if proposal_failed else np.ones((2, 24)) / 24, block_keys=block_keys,
            setup_seconds=.25, query_seconds=total - .25, total_cold_seconds=total,
            setup_costs={}, query_costs=costs, cumulative_costs=costs, batch_dispatches={},
            exact_frequency_strategies={}, exact_expected_rhs=None)
    monkeypatch.setattr(development, "execute_control", control)
    output = tmp_path / f"mock-{failure_mode}"
    summary = development.run_development(PROTOCOL, output, root=ROOT)
    assert summary["n_cold_calls"] == 288 and summary["n_block_quality_rows"] == 6912
    assert summary["checks"]["complete_fixed_cohort"] and summary["checks"]["repeat_parity"]
    if failure_mode is not None:
        assert summary["status"] == "checks-failed"
        assert summary["checks"]["finite_objectives_and_gradients"] is (failure_mode != "proposal")
        assert not summary["checks"]["no_numeric_certificate_failures"]
        assert summary["development_recommendation"]["recommended_development_candidate"] is None
    else:
        assert summary["status"] == "complete"
        assert summary["development_recommendation"]["recommended_development_candidate"] == "corrected_batch64"
    with np.load(output / "aggregate_vectors.npz", allow_pickle=False) as vectors:
        assert vectors["directions"].shape == (288, 2)
        expected_gradients = 240 if failure_mode == "proposal" else 288
        assert vectors["gradients"].shape == (expected_gradients, 2)
        assert len(vectors["gradient_row_indices"]) == expected_gradients
