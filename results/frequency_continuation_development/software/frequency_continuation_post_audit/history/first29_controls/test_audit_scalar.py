"""Pure/manufactured audit controls; no corpus NPZ, PDE or optimizer calls."""
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_frequency_scalar_audit_controls", HERE/"audit_scalar.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def test_default_cli_cannot_open_a_project_data_entrypoint(monkeypatch, capsys):
    monkeypatch.setattr(A, "audit_execute", lambda *a: pytest.fail("no execute on default CLI"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no project numeric decode"))
    assert A.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "metadata_only_no_project_arrays_or_solves"


def protocol():
    return dict(identities=[dict(case=i) for i in range(8)])


def test_full_schedule_all96_24quadruples_32groups_not_success_subset():
    planned = A.expected_schedule(protocol())
    checks = A.Checks()
    A.schedule_check(planned, planned, checks, "all")
    assert len(planned) == 96 and checks.issues == []
    A.schedule_check(planned[:-1], planned, checks, "missing")
    assert checks.issues


def timing_row(wall=7.8, *, inner_eligible=True, method_valid=True):
    started, inner = 10., 7.7
    returned = started+wall
    wall = returned-started
    last = 17.6 if inner_eligible else 18.2
    trace = dict(global_started_perf_counter=started, last_observed_absolute_clock=last,
        method_valid_before_return_gate=method_valid)
    inner_ok = inner_eligible and method_valid
    eligible = inner_ok and wall <= 8.
    row = dict(caller_observed_return_stamp=returned, timing=dict(
        actual_api_return_seconds=wall, inner_execute_timed_return_seconds=inner,
        measured_helper_return_tail_seconds=wall-inner, api_return_cap_seconds=8.,
        overshoot_seconds=max(0., wall-8.), runtime_eligible=eligible,
        event_seconds=wall if eligible else None,
        inner_runtime_gate=dict(eligible=inner_ok, event_seconds=inner if inner_ok else None)))
    return row, trace


@pytest.mark.parametrize("wall,inner,expected", [(8., True, True), (8.+1e-10, True, False),
    (7.8, False, False), (8.1, False, False)])
def test_strict_outer8_and_inner_invalid_never_rescued(wall, inner, expected):
    row, trace = timing_row(wall, inner_eligible=inner)
    checks = A.Checks()
    eligible, _ = A.runtime_check(row, trace, checks, "timing")
    assert eligible is expected and checks.issues == []


def test_forged_outer_eligibility_with_invalid_inner_is_detected():
    row, trace = timing_row(inner_eligible=False)
    row["timing"]["runtime_eligible"] = True
    checks = A.Checks()
    assert A.runtime_check(row, trace, checks, "timing")[0] is False
    assert checks.issues


def prep_counts():
    return dict(factorizations=8, forward_rhs=48, tangent_rhs=0, adjoint_rhs=0,
        stability_rhs=0, dense_inverse_rhs=0, svd_evaluations=0, cache_hits=0, analytic_bound_evaluations=0)


def cost_trace():
    costs = dict(native=dict(physics=dict(factorizations=8, forward_rhs=48), oracle=dict(queries=2)), delegate=dict(evaluate_attempts=2))
    return dict(owners=[dict(label="stage1", latest_costs=costs, costs_complete=True)],
        preparation_costs=dict(all8_observation_generation=dict(latest_counts=prep_counts(), complete=True)),
        in_flight_work_unknown=False, preparation_in_flight_counts_unknown=False, costs_complete=True,
        owner_costs_summed_once=copy.deepcopy(costs), preparation_costs_separate=prep_counts(),
        owner_known_counts_partial_sum=None, preparation_known_counts_partial_sum=None)


def test_costs_count_owner_once_and_preparation_separately():
    trace, checks = cost_trace(), A.Checks()
    owner, prep = A.cost_check(trace, checks, "costs")
    assert checks.issues == [] and owner["native"]["physics"]["factorizations"] == 8
    assert prep["factorizations"] == 8  # Not16 hidden in either separate category.
    trace["owner_costs_summed_once"]["native"]["physics"]["factorizations"] = 16
    A.cost_check(trace, checks, "doublecharge")
    assert checks.issues
    trace, checks = cost_trace(), A.Checks()
    trace["owners"].append(dict(trace["owners"][0]))
    A.cost_check(trace, checks, "duplicatedowner")
    assert checks.issues


def test_incomplete_or_invalid_preflight_keeps96_without_numeric_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("incomplete/invalid preflight must never decode numeric data")
    monkeypatch.setattr(A, "load_references", forbidden)
    monkeypatch.setattr(A, "row_audit", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    planned = A.expected_schedule(protocol())
    rows = [dict(r, status="complete" if i < 85 else "attempted" if i == 85 else "unattempted",
        final_study_eligible=i < 85) for i, r in enumerate(planned)]
    references, records = A.collect_records(planned, rows, None, None, protocol(), {}, A.Checks(), False)
    assert references == [] and len(records) == 96
    assert all(r["metadata_only"] and not r["numeric_data_decoded"] for r in records)
    result = A.summarize(records, False)
    assert result["full_run_denominator"] == 96 and len(result["groups"]) == 32
    assert sum(v["recorded_status_counts"].get("attempted", 0) for v in result["arms"].values()) == 1
    assert all(v["verified_eligible"] == 0 for v in result["arms"].values())


def test_unknown_owner_cost_is_null_with_known_partial_sum_not_zero():
    trace, checks = cost_trace(), A.Checks()
    known = trace["owner_costs_summed_once"]
    trace["owners"].append(dict(label="stage2", latest_costs=None, costs_complete=False))
    trace.update(costs_complete=False, owner_costs_summed_once=None, owner_known_counts_partial_sum=known)
    A.cost_check(trace, checks, "unknown")
    assert checks.issues == []
    trace["owner_costs_summed_once"] = {}
    A.cost_check(trace, checks, "inventedzero")
    assert checks.issues


def test_boolean_is_not_a_counter_or_a_return_time():
    with pytest.raises(ValueError):
        A.count_tree(dict(work=True))
    row, trace = timing_row()
    row["caller_observed_return_stamp"] = True
    checks = A.Checks()
    assert A.runtime_check(row, trace, checks, "bool")[0] is False and checks.issues


@pytest.mark.parametrize("exit_code,all_passed,valid", [(0, True, True), (2, False, True),
    (2, True, False), (0, False, False), (-15, True, False), (7, False, False), (True, True, False)])
def test_actual_exit_matches_pinned_main_contract(exit_code, all_passed, valid):
    checks = A.Checks()
    assert A.complete_exit_check(exit_code, dict(numerical_audits_all_pass=all_passed), checks) is valid
    assert bool(checks.issues) is not valid


def test_complete_cost_flag_cannot_hide_absent_owner_snapshot():
    trace, checks = cost_trace(), A.Checks()
    trace["owners"][0]["latest_costs"] = None
    trace["owner_costs_summed_once"] = {}
    A.cost_check(trace, checks, "missing_snapshot")
    assert any(i["where"].endswith("completed_owner_requires_snapshot") for i in checks.issues)


def summary(records, *, pending=None, omitted=0, all_count=None):
    finalized = records if pending is None else records[:-1]
    return dict(total_count=len(records)+omitted if all_count is None else all_count,
        retained_count=len(finalized), omitted_count=omitted,
        finalized_count=len(finalized)+omitted, chain_sha256=A.record_chain(finalized),
        head_limit=8, tail_limit=8, pending_record=pending,
        pending_record_unsealed=pending is not None)


def test_complete_scalar_stream_chain_and_corruption():
    records = [dict(index=i, model_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(3)]
    metadata, checks = summary(records), A.Checks()
    assert A.stream_check(records, metadata, checks, "stream") and not checks.issues
    metadata["chain_sha256"] = "0"*64
    A.stream_check(records, metadata, checks, "badchain")
    assert checks.issues


def test_unsealed_prefix_is_not_completed_and_omitted_middle_not_replayed():
    records, checks = [dict(event=1), dict(event=2)], A.Checks()
    assert A.stream_check(records, summary(records, pending=records[-1]), checks, "pending") is False
    assert checks.issues == []
    # A valid-shaped commitment with an omitted middle is only structural
    # evidence; the audit deliberately cannot reconstruct its hash from samples.
    metadata = summary(records, omitted=4)
    metadata["chain_sha256"] = "a"*64
    assert A.stream_check(records, metadata, checks, "omitted") and checks.issues == []


def manufactured_trace(method):
    """Serialized scalar descriptors only; no oracle/controller/engine import."""
    protocol_data = dict(physics=dict(shape=[49, 49], spacing=.025, damping_width=6, damping_strength=2.),
        controller_settings=dict(control="fixed_fixture"), production_source_sha256="9"*64,
        local_dependency_sha256={"bounded_exact.py": "d"*64})
    h = "a"*64
    trace = dict(arm=method, settings=protocol_data["controller_settings"], stages=[], owners=[],
        available8_data_sha256=h, available_frequencies_hz=list(A.ALL8), bounds_m=[1/36, 1.],
        truth_input=False, checkpoint_vectors_retained=False, initial_model_sha256=h,
        final_model_sha256=h, status="completed_paid_state", costs_complete=True,
        in_flight_work_unknown=False, method_valid_before_return_gate=True,
        source_binding=dict(production_sha256="9"*64, engine_source_sha256="d"*64),
        common_initial_full8=dict(model_sha256=h, objective=2., geometry_sha256=h, data_sha256=h),
        common_final_full8=dict(model_sha256=h, objective=2., geometry_sha256=h, data_sha256=h))
    for index, frequencies in enumerate(A.STAGES[method]):
        age = .5+index
        checkpoints = [dict(kind="initial", model_sha256=h, objective=2., elapsed_seconds=age, within_budget=True)]
        engine = dict(checkpoints=checkpoints, checkpoint_trace=summary(checkpoints), steps=[], step_trace=summary([]),
            initial_model_sha256=h, final_model_sha256=h, final_objective=2., status="maximum_steps",
            costs_complete=True, native_callback_invocations=0, native_callback_point_records=0,
            native_callback_count=0, last_native_callback_attempt=None,
            all_library_evaluation_trace=dict(total_count=0, chain_sha256=A.record_chain([])),
            scipy_maxfun_default=15000, scipy_maxfun_passed_explicitly=False)
        trace["stages"].append(dict(index=index, frequencies_hz=list(frequencies), maximum_steps=2048 if len(A.STAGES[method]) == 2 else 4096,
            absolute_deadline_seconds=3.75 if len(A.STAGES[method]) == 2 and index == 0 else 7.5,
            engine_result=engine, status="maximum_steps", start_model_sha256=h))
        trace["owners"].append(dict(label=f"stage{index+1}", role="policy", frequencies_hz=list(frequencies),
            status="constructed_cold", contract=dict(operator_base_record=dict(configuration=protocol_data["physics"]),
                reference_source_sha256="caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131",
                operator_base_sha256=h, geometry_sha256=h, data_sha256=h, survey_sha256=h)))
        trace["final_model_origin"] = dict(stage_index=index, model_sha256=h, elapsed_seconds=age)
    if method in {"low4_then8", "original4"}:
        metric = copy.deepcopy(trace["owners"][0])
        metric.update(label="common8_metric", role="metric_only", frequencies_hz=list(A.ALL8))
        trace["owners"].insert(0, metric)
    return trace, protocol_data


@pytest.mark.parametrize("method", A.METHODS)
def test_serialized_four_arm_receipt_chronology_and_method_failure(method):
    trace, protocol_data = manufactured_trace(method)
    checks = A.Checks()
    assert A.trace_check(trace, method, protocol_data, checks, "fixture")["method_valid"]
    assert checks.issues == []
    trace["stages"][0]["engine_result"].update(status="failed", costs_complete=False)
    trace.update(status="failed_method", method_valid_before_return_gate=False)
    checks = A.Checks()
    result = A.trace_check(trace, method, protocol_data, checks, "failed")
    assert not result["method_valid"] and result["method_failure"] and checks.issues == []


def test_forged_missing_stage_owner_cannot_earn_serialized_credit():
    trace, protocol_data = manufactured_trace("joint8")
    trace["owners"] = []
    checks = A.Checks()
    A.trace_check(trace, "joint8", protocol_data, checks, "missing_owner")
    assert any(i["where"].endswith("actual_policy_owner_required") for i in checks.issues)


def test_manual_halfcell_edge_extension_and_scalar_model_units():
    raw = np.full((24, 24), .25)
    velocity, truth = A.manual_refinement(raw)
    assert len(velocity) == len(truth) == 2401 and all(v == 2. for v in velocity)
    assert all(m == .25 for m in truth)
    own = A.scalar_quality(np.full(2401, .24), truth)
    assert own["relative_squared_slowness_L2"] == pytest.approx(.04, abs=1e-15)
    assert own["relative_physical_velocity_L2"] == pytest.approx((1/math.sqrt(.24)-2)/2, abs=1e-15)
    vzx = np.fromfunction(lambda z, x: 2.+.02*z+.03*x, (24, 24))
    fine, _ = A.manual_refinement(1/(vzx*vzx))
    assert fine[0] == pytest.approx(vzx[0, 0], abs=1e-14)
    assert fine[2*49+2] == pytest.approx(.25*(vzx[0, 0]+vzx[0, 1]+vzx[1, 0]+vzx[1, 1]), abs=1e-14)


def records96():
    return [dict(r, recorded_status="complete", local_credit_conditions_pass=True,
        own_quality=dict(final=dict(relative_squared_slowness_L2=.1, relative_physical_velocity_L2=.2)),
        verified_credit_eligible=False, verified_quality=None, common_full8_ratio=.2,
        actual_api_return_seconds=7.8, targets=[dict(ratio=.5, returned_ratio_reached=True), dict(ratio=.25, returned_ratio_reached=True)])
        for r in A.expected_schedule(protocol())]


def test_full3_medians_missing_failure_denominators_and_global_fatal_gate():
    rows = records96()
    rows[0]["local_credit_conditions_pass"] = False
    rows[0]["recorded_status"] = "failed"
    result = A.summarize(rows, True)
    assert len(result["groups"]) == 32 and result["full_run_denominator"] == 96
    group = next(g for g in result["groups"] if g["case_index"] == rows[0]["case_index"] and g["method"] == rows[0]["method"])
    assert group["verified_repetitions"] == 2 and all(v is None for v in group["median_quality"].values())
    assert all(a["planned_runs"] == 24 for a in result["arms"].values())
    fatal = A.summarize(records96(), False)
    assert all(a["verified_eligible"] == 0 for a in fatal["arms"].values())
    assert all(v is None for g in fatal["groups"] for v in g["median_quality"].values())


def test_missing_manifest_file_and_path_escape_are_integrity_failures(tmp_path):
    checks = A.Checks()
    A.manifest_check(tmp_path, dict(artifact_sha256={"missing.json": "0"*64}), checks)
    assert checks.issues
    assert A.safe_file(tmp_path, "../escape", checks, "escape") is None
