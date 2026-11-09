"""Tiny manufactured scalars and synthetic24x49 arrays only; no project/PDE."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import audit_retained as a
import pytest

HERE = Path(__file__).resolve().parent
CRITERIA = json.loads((HERE / "criteria.json").read_bytes())
PROTOCOL = json.loads(
    (HERE.parent / "fractional_velocity_prior48_driver_candidate/protocol.json").read_bytes()
)


def test_fixed_criteria_hash_and_no_timing_tolerance():
    assert a.file_sha(HERE / "criteria.json") == a.CRITERIA_SHA
    assert CRITERIA["engine_deadline_seconds"] == 7.5
    assert CRITERIA["complete_return_cap_seconds"] == 8.0
    assert CRITERIA["timing_tolerance_seconds"] == 0.0
    for name in (
        "prior_rtol",
        "prior_atol",
        "composed_F_rtol",
        "composed_F_atol",
        "objective_rtol",
        "objective_atol_initial_scale",
        "objective_atol_initial_floor",
        "observation_rtol",
    ):
        assert CRITERIA[name] == PROTOCOL["audit_tolerances"][name]


def test_constant_halfindex_refinement_and_scalar_quality():
    truth, fine = a.scalar_refinement([[0.25] * 24 for _ in range(24)])
    assert truth == [0.25] * 2401 and all(v == 2.0 for row in fine for v in row)
    errors = a.quality([0.25] * 2401, truth)
    assert errors == dict(relative_squared_slowness_L2=0.0, relative_physical_velocity_L2=0.0)
    errors = a.quality([1.0] * 2401, truth)
    assert errors["relative_squared_slowness_L2"] == pytest.approx(3.0)
    assert errors["relative_physical_velocity_L2"] == pytest.approx(0.5)


def test_scalar_bilinear_halfpoint_is_velocity_not_mass_average():
    raw = [[0.25] * 24 for _ in range(24)]
    raw[0][1] = 1.0
    truth, fine = a.scalar_refinement(raw)
    # fine row1 is raw depth0; col2 half between rawx0 andx1.
    assert fine[1][2] == 1.5
    assert truth[49 + 2] == pytest.approx(1 / 2.25)
    assert truth[49 + 2] != (0.25 + 1) / 2


def test_constant_update_and_corner_prior_degree():
    initial = [0.25] * 6
    assert 100 * a.scalar_R([0.25 / 1.1**2] * 6, initial, (2, 3), 0.025, 0.1) == pytest.approx(0.5)
    corner = initial.copy()
    corner[0] /= 1.2**2
    assert a.scalar_R(corner, initial, (2, 3), 0.025, 0.1) == pytest.approx(
        0.2**2 * (1 + 2 * 16) / 12
    )


def test_model_hash_is_raw_little_endian_float64():
    assert a.model_sha([0.25, 0.5]) == hashlib.sha256(struct.pack("<dd", 0.25, 0.5)).hexdigest()


def make_fixture(weight=100.0):
    p = copy.deepcopy(PROTOCOL)
    initial, final = [0.25] * 2401, [0.26] * 2401
    raw = [[0.25] * 24 for _ in range(24)]
    truth, _ = a.scalar_refinement(raw)
    raw_phi0, raw_phi = 1.0, 0.7
    R = a.scalar_R(final, initial, (49, 49), 0.025, 0.1) if weight else 0.0
    F = raw_phi + weight * R
    geometry = a.json_sha(a.expected_geometry(p))

    def header(model, phi):
        operator = a.json_sha({"manufactured_operator_for": a.model_sha(model)})
        data = "c" * 64
        return dict(
            objective=phi,
            model_sha256=a.model_sha(model),
            operator_sha256=operator,
            geometry_sha256=geometry,
            data_sha256=data,
            key=a.json_sha(dict(operator=operator, geometry=geometry, data=data)),
            gradient_cached=False,
            frequency_merit_routes={str(float(f)): "forward_states" for f in p["frequencies_hz"]},
            frequency_field_shapes={str(float(f)): [2401, 6] for f in p["frequencies_hz"]},
        )

    raw0, rawfinal = header(initial, raw_phi0), header(final, raw_phi)
    initial_header = dict(
        objective=1.0,
        normalized_data_value=1.0,
        weighted_prior_value=0.0,
        native_data_receipt_header=raw0,
        model_sha256=a.model_sha(initial),
    )
    current = dict(
        objective=F,
        normalized_data_value=0.7,
        weighted_prior_value=weight * R,
        native_data_receipt_header=rawfinal,
        model_sha256=a.model_sha(final),
    )
    counts = dict(
        physics=dict(factorizations=16, forward_rhs=96, adjoint_rhs=48),
        oracle=dict(merit_evaluations=2),
    )
    engine = dict(
        adapter=dict(
            weight=weight,
            initial_accepted=initial_header,
            current_accepted=current,
            initial_paid_Phi0=1.0,
            adapter_counts=dict(prior_evaluations_completed=int(weight > 0)),
        ),
        total_online_costs=dict(oracle=dict(native=counts)),
        final_objective=F,
    )
    payload = dict(
        initial_model=initial,
        final_model=final,
        engine=engine,
        weight=weight,
        physical_acceptance=None,
        no_certificate_claim=True,
        method_valid=True,
        initial_native_header=initial_header,
        final_raw_native_header=rawfinal,
        final_R=R if weight else None,
        final_F=F,
        common_raw_Phi_ratio=0.7,
        final_native_owner_cumulative_costs=counts,
        metric_tail_counts=dict(
            native_snapshot_calls=1,
            native_evaluate_attempts=1,
            native_evaluate_completed=1,
            prior_evaluation_attempts=int(weight > 0),
            prior_evaluations_completed=int(weight > 0),
            final_metric_prior_anchor_constructions=int(weight > 0),
        ),
    )
    records = {}
    for name, model, phi in [("initial_model", initial, 1.0), ("final_model", final, 0.7)]:
        records[name] = dict(
            model_sha256=a.model_sha(model),
            objective=phi,
            per_frequency_objectives=[phi / 8] * 8,
            **a.quality(model, truth),
        )
    graph = dict(
        records=records,
        failures=[],
        factorizations=24,
        forward_rhs=144,
        maximum_relative_linear_residual=1e-14,
        numeric_comparisons=3 + int(weight > 0),
        checks=6 * 24 + 14 + int(weight > 0),
    )
    row = dict(
        run_index=0,
        case_index=0,
        repetition=0,
        method=a.ARMS[int(weight > 0)],
        weight=weight,
        common_binding=dict(
            initial_native_m_sha256=a.model_sha(initial),
            raw_model_sha256=a.model_sha([0.25] * 576),
            truth_sha256=a.model_sha(truth),
            all8_observations_sha256="d" * 64,
        ),
        timing=dict(
            actual_complete_return_seconds=7.7,
            inner_return_age=7.69,
            complete_return_cap_seconds=8.0,
            eligible=True,
            event_seconds=7.7,
        ),
        integrity_valid=True,
        preparation_costs=dict(factorizations=8, forward_rhs=48),
        preparation_costs_unknown=False,
        common_raw_Phi_ratio=0.7,
        initial_quality=a.quality(initial, truth),
        primary_quality=a.quality(final, truth),
        final_study_eligible=True,
    )
    return row, payload, graph, raw, p


def fake_trace(engine, checks, criteria, label):
    return dict(
        method_valid=True,
        last_eligible=dict(
            model_sha256=engine["adapter"]["current_accepted"]["model_sha256"],
            objective=engine["final_objective"],
        ),
        checkpoint_omitted=0,
        step_omitted=0,
    )


def inspect_fixture(fixture):
    row, payload, graph, raw, p = fixture
    checks = a.Checks()
    result = a.audit_row(
        row,
        payload,
        graph,
        raw,
        p,
        CRITERIA,
        checks,
        fake_trace,
        initial_builder=lambda fine: [0.25] * 2401,
    )
    return result, checks


@pytest.mark.parametrize("weight", [0.0, 100.0])
def test_manufactured_whole_row_passes_both_normalized_arms(weight):
    result, checks = inspect_fixture(make_fixture(weight))
    assert not checks.failures, checks.failures
    assert result["eligible"] is True


@pytest.mark.parametrize(
    "kind",
    [
        "R",
        "F",
        "native_header",
        "data_header",
        "Phi0",
        "quality",
        "initializer",
        "count_boolean",
        "preparation_RHS",
        "graph_Phi",
        "graph_work",
        "selected_vector",
    ],
)
def test_manufactured_scalar_header_counter_corruption_rejected(kind):
    fixture = make_fixture()
    row, payload, graph, _raw, _p = fixture
    if kind == "R":
        payload["final_R"] += 1.0
    elif kind == "F":
        payload["final_F"] += 1.0
    elif kind == "native_header":
        payload["final_raw_native_header"]["model_sha256"] = "0" * 64
    elif kind == "data_header":
        payload["final_raw_native_header"]["data_sha256"] = "0" * 64
    elif kind == "Phi0":
        payload["engine"]["adapter"]["initial_paid_Phi0"] = 2.0
    elif kind == "quality":
        row["primary_quality"]["relative_physical_velocity_L2"] += 1.0
    elif kind == "initializer":
        payload["initial_model"][0] = 0.251
    elif kind == "count_boolean":
        payload["final_native_owner_cumulative_costs"]["oracle"]["merit_evaluations"] = True
    elif kind == "preparation_RHS":
        row["preparation_costs"]["forward_rhs"] = 47
    elif kind == "graph_Phi":
        graph["records"]["final_model"]["objective"] += 1.0
    elif kind == "graph_work":
        graph["forward_rhs"] = 999
    else:
        payload["final_model"][0] = 0.261
    result, checks = inspect_fixture(fixture)
    assert checks.failures
    assert result["eligible"] is False
    assert result["final_quality"] is None


def test_strict8_second_eligibility_has_no_epsilon():
    fixture = make_fixture()
    row, *_ = fixture
    row["timing"].update(
        actual_complete_return_seconds=8.000000000001,
        inner_return_age=8.0,
        eligible=False,
        event_seconds=None,
    )
    row.update(final_study_eligible=False, primary_quality=None)
    result, checks = inspect_fixture(fixture)
    assert not checks.failures, checks.failures
    assert result["eligible"] is False and result["final_quality"] is None


def summary_rows(candidate=0.08, baseline=0.1):
    rows = []
    for case in range(8):
        for method in a.ARMS:
            for rep in range(3):
                rows.append(
                    dict(
                        case_index=case,
                        method=method,
                        repetition=rep,
                        eligible=True,
                        initial_quality=dict(relative_physical_velocity_L2=0.1),
                        final_quality=dict(
                            relative_physical_velocity_L2=baseline
                            if method == a.ARMS[0]
                            else candidate
                        ),
                        common_raw_Phi_ratio=0.5,
                    )
                )
    return rows


def test_all48_16groups8pairs_and_original_recovery_gates():
    value = a.summarize(summary_rows(), CRITERIA)
    assert (
        value["all48_eligible"] is True
        and len(value["case_arm_groups"]) == 16
        and len(value["cases"]) == 8
    )
    assert value["proposed_development_continuation_gate_pass"] is True
    assert (
        a.summarize(summary_rows(candidate=0.1, baseline=0.7), CRITERIA)[
            "proposed_development_continuation_gate_pass"
        ]
        is False
    )


@pytest.mark.parametrize("kind", ["late", "missing", "duplicate", "harm", "fit", "only5_success"])
def test_denominators_harm_fit_and_success_count_not_relaxed(kind):
    rows = summary_rows()
    if kind == "late":
        rows[0].update(eligible=False, final_quality=None)
    elif kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif kind == "harm":
        for r in rows:
            if r["case_index"] == 7 and r["method"] == a.ARMS[1]:
                r["final_quality"]["relative_physical_velocity_L2"] = 0.12
    elif kind == "fit":
        for r in rows:
            if r["method"] == a.ARMS[1]:
                r["common_raw_Phi_ratio"] = 0.75000000001
    else:
        for r in rows:
            if r["case_index"] >= 5 and r["method"] == a.ARMS[1]:
                r["final_quality"]["relative_physical_velocity_L2"] = 0.1
    value = a.summarize(rows, CRITERIA)
    assert value["proposed_development_continuation_gate_pass"] is False


def test_default_metadata_does_not_read_project_or_run(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["audit_retained.py"])
    monkeypatch.setattr(a, "read_cases", lambda *args: pytest.fail("project arrays read"))
    monkeypatch.setattr(a, "bind_run", lambda *args: pytest.fail("run artifacts read"))
    assert a.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "METADATA_ONLY_NO_PROJECT_ARRAYS_RUN_OR_PDE"
    )


@pytest.mark.parametrize("corrupt", ["no_terminal", "wrong_run", "Boolean_exit"])
def test_root_terminal_gate_precedes_run_read(tmp_path, monkeypatch, corrupt):
    from types import SimpleNamespace

    run = tmp_path / "future_run"
    repo = tmp_path / "repo"
    observation = dict(observed_terminal=True, run_directory=str(run.resolve()), actual_exit_code=0)
    if corrupt == "no_terminal":
        observation["observed_terminal"] = False
    elif corrupt == "wrong_run":
        observation["run_directory"] = "wrong"
    else:
        observation["actual_exit_code"] = False
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps(observation))
    args = SimpleNamespace(
        run=run,
        repo=repo,
        terminal_observation=terminal,
        expected_terminal_sha256=a.file_sha(terminal),
    )
    monkeypatch.setattr(
        a, "safe_file", lambda *args: pytest.fail("run read before terminal admission")
    )
    with pytest.raises(ValueError):
        a.execute(args, CRITERIA)


def test_nonfinite_corruption_is_reported_without_silent_zero_imputation():
    checks = a.Checks()
    checks.near(float("nan"), 1.0, rtol=1e-9, atol=1e-12, label="bad")
    assert checks.failures and checks.failures[0]["detail"]["independent"] == "nan"
    assert json.loads(json.dumps(a.clean_json(dict(bad=float("inf"))), allow_nan=False)) == dict(
        bad="inf"
    )


@pytest.mark.parametrize("weight", [0.0, 100.0])
def test_main_with_real_independent_trace_checker_on_manufactured_row(weight):
    import test_trace_checks as t
    import trace_checks

    fixture = make_fixture(weight)
    row, payload, graph, raw, p = fixture
    e = t.engine()
    initial_sha = a.model_sha(payload["initial_model"])
    final_sha = a.model_sha(payload["final_model"])
    F = payload["final_F"]
    e["checkpoints"][0]["model_sha256"] = initial_sha
    e["steps"][0]["model_sha256"] = initial_sha
    e["checkpoints"][-1].update(model_sha256=final_sha, objective=F)
    step = e["steps"][-1]
    step.update(objective_after=F)
    step["native_callback"].update(candidate_sha256=final_sha, objective=F)
    for event in step["library_evaluations"]:
        event["candidate_sha256"] = final_sha
        if event["kind"] == "merit":
            event["objective"] = F
    e["last_native_callback_attempt"]["candidate_sha256"] = final_sha
    e["final_objective"] = F
    e["adapter"] = copy.deepcopy(payload["engine"]["adapter"])
    e["adapter"].update(initial_native_m_sha256=initial_sha, latest_native_snapshot=dict(queries=8))
    e["adapter"]["adapter_counts"]["queries"] = 8
    e["total_online_costs"]["oracle"]["adapter"] = e["adapter"]["adapter_counts"].copy()
    for point in (e["adapter"]["initial_accepted"], e["adapter"]["current_accepted"]):
        point.update(
            kind="classical_composed_objective_point",
            native_receipt_fabricated=False,
            certificate_claim=False,
        )
    payload["initial_native_header"] = e["adapter"]["initial_accepted"]
    payload["engine"] = e
    payload["final_native_owner_cumulative_costs"] = dict(queries=9)
    t.seal(e)
    checks = a.Checks()
    result = a.audit_row(
        row,
        payload,
        graph,
        raw,
        p,
        CRITERIA,
        checks,
        trace_checks.audit_engine,
        initial_builder=lambda fine: [0.25] * 2401,
    )
    assert not checks.failures, checks.failures
    assert result["eligible"] is True


def test_relocated_copy_uses_local_hash_bound_directory_not_original_runner_path(tmp_path):
    local = tmp_path.resolve() / "extracted_run"
    terminal = dict(
        observed_terminal=True,
        run_directory=str(local),
        actual_exit_code=0,
        original_executed_run_directory="/home/runner/work/frozen-repo/private_run",
        remote_run_id=123,
        remote_job_id=456,
        child_exit_receipt_sha256="a" * 64,
    )
    context = a.terminal_context(terminal, local)
    assert context["audited_run_directory"] == str(local)
    assert context["original_executed_run_directory"] != str(local)
    assert context["root_reported_remote_execution_context"]["remote_job_id"] == 456
    assert (
        context["root_reported_remote_execution_context"]["child_exit_receipt_sha256"] == "a" * 64
    )


@pytest.mark.parametrize("field", ["negative_graph_norm", "negative_initial_norm"])
def test_impossible_negative_norms_do_not_pass_within_numeric_tolerance(field):
    fixture = make_fixture()
    row, payload, graph, raw, p = fixture
    if field == "negative_graph_norm":
        graph["records"]["initial_model"]["relative_physical_velocity_L2"] = -1e-15
    else:
        row["initial_quality"]["relative_physical_velocity_L2"] = -1e-15
    result, checks = inspect_fixture(fixture)
    assert checks.failures and result["eligible"] is False
    assert any("nonnegative" in issue["label"] for issue in checks.failures)


def test_cross_arm_data_binding_mismatch_invalidates_both_case_arms():
    rows = summary_rows()
    for row in rows:
        row.update(
            common_binding=dict(initial="same", observations="same"),
            saved_native_data_sha256="a" * 64,
        )
    rows[0]["common_binding"]["observations"] = "changed"
    checks = a.Checks()
    assert a.bind_case_results(rows, checks, 0) is False
    assert checks.failures
    assert all(
        r["eligible"] is False and r["final_quality"] is None for r in rows if r["case_index"] == 0
    )
    assert a.summarize(rows, CRITERIA)["proposed_development_continuation_gate_pass"] is False


def test_any_global_integrity_issue_prevents_positive_study_or_gate_credit():
    before = a.summarize(summary_rows(), CRITERIA)
    assert before["proposed_development_continuation_gate_pass"] is True
    checks = a.Checks()
    checks.require(False, "artifact changed after audit")
    after = a.apply_global_integrity_gate(before, checks)
    assert after["all48_eligible"] is False
    assert after["proposed_development_continuation_gate_pass"] is False
    assert after["global_integrity_gate_pass"] is False
