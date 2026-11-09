"""Metadata, paid-return, failure-prefix and manufactured API controls only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "wavecert-fwi"
SOURCE_CONTROLLER = HERE.with_name("frequency_continuation_controller")
spec = importlib.util.spec_from_file_location("_frequency_driver_controls", HERE / "run_study.py")
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


def protocol():
    return driver.load_protocol(HERE / "protocol.json")


def test_default_is_metadata_only_and_cannot_call_runtime(monkeypatch, capsys):
    monkeypatch.setattr(driver, "load_runtime", lambda *a: pytest.fail("no runtime on default CLI"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no project arrays"))
    assert driver.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "metadata_only_no_project_arrays_or_solves"
    assert output["planned_runs"] == 96


def test_full_four_arm_schedule_and_no_input_state():
    p, _ = protocol()
    rows = driver.schedule(p)
    assert len(rows) == 96 and rows == driver.schedule(p)
    for repetition in range(3):
        for case in range(8):
            block = [r for r in rows if r["repetition"] == repetition and r["case_index"] == case]
            assert len(block) == 4 and {r["method"] for r in block} == set(driver.METHODS)
            assert all(r["status"] == "unattempted" and r["primary_quality"] is None for r in block)


def test_fixed_raw_schema_and_explicit_eight_frequency_preparation_are_separate():
    p, old = protocol()
    untouched = copy.deepcopy(old)
    preparation = driver.preparation_protocol(old, p)
    assert old == untouched and old["geometry"]["frequencies_hz"] == [2.5, 3., 3.5, 4.]
    assert preparation["geometry"]["frequencies_hz"] == driver.FREQUENCIES
    assert preparation["geometry"]["blocks_per_model"] == 48
    preparation["geometry"]["frequencies_hz"] = old["geometry"]["frequencies_hz"]
    preparation["geometry"]["blocks_per_model"] = 24
    assert preparation == old


@pytest.mark.parametrize("inner_eligible,outer_wall,expected", [(True, 7.8, True), (True, 8.01, False),
    (False, 7.8, False), (False, 8.01, False)])
def test_outer_return_tail_is_paid_and_cannot_rescue_inner_invalid(inner_eligible, outer_wall, expected):
    measured = dict(payload=dict(method_valid_before_return_gate=True),
        timing=dict(global_started=10., actual_api_return_seconds=7.7),
        runtime_gate=dict(eligible=inner_eligible, reason="manufactured"))
    result = driver.outer_timing(measured, 10.+outer_wall, 8.)
    assert result["runtime_eligible"] is expected
    assert result["actual_api_return_seconds"] == pytest.approx(outer_wall)
    assert result["measured_helper_return_tail_seconds"] == pytest.approx(outer_wall-7.7)
    assert result["event_seconds"] == pytest.approx(outer_wall) if expected else result["event_seconds"] is None


def test_unknown_dependency_review_cannot_execute(tmp_path):
    p, _ = protocol()
    p["controller_dependencies_pending_review"] = True
    for name in p["local_dependency_sha256"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, tmp_path / name)
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    with pytest.raises(driver.StaleFreeze, match="not frozen"):
        driver.load_protocol(tmp_path / "protocol.json", executable=True)


def test_scalar_trace_does_not_flatten_unexpected_field_arrays():
    with pytest.raises(TypeError):
        driver.scalar_trace(dict(initial_model=np.ones(3), final_model=np.ones(3),
            unexpected_full_state=np.ones((2, 3))))


def test_interrupted_paid_prefix_vectors_saved_without_quality_credit(tmp_path):
    initial, final = np.full(2401, .2), np.full(2401, .19)
    payload = dict(initial_model=initial, final_model=final, status="interrupted",
        method_valid_before_return_gate=False, in_flight_work_unknown=True)
    row = dict(run_index=0, status="interrupted", final_study_eligible=False, primary_quality=None)
    driver.failure_witness(tmp_path, row, dict(payload=payload), {}, SimpleNamespace(np=np))
    assert row["failure_prefix_saved"] is True and row["primary_quality"] is None
    assert row["final_study_eligible"] is False
    with np.load(tmp_path / row["models_artifact"], allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["initial_model"], initial)
        np.testing.assert_array_equal(bundle["final_model"], final)


def test_failure_saving_error_is_recorded_and_never_overwrites(tmp_path):
    existing = tmp_path / "run_000_failure_prefix_models.npz"
    existing.write_bytes(b"manufactured preexisting data")
    row = dict(run_index=0, status="interrupted", primary_quality=None)
    payload = dict(initial_model=np.ones(2401), final_model=np.ones(2401), status="interrupted")
    driver.failure_witness(tmp_path, row, dict(payload=payload), {}, SimpleNamespace(np=np))
    assert row["failure_prefix_saved"] is False and "FileExistsError" in row["failure_prefix_save_error"]
    assert existing.read_bytes() == b"manufactured preexisting data"
    assert row["status"] == "interrupted"


def test_scalar_failure_prefix_does_not_claim_unavailable_vectors(tmp_path):
    row = dict(run_index=0, status="failed", primary_quality=None)
    payload = dict(initial_model=None, final_model=None, status="failed_controller")
    driver.failure_witness(tmp_path, row, dict(payload=payload), {}, SimpleNamespace(np=np))
    assert row["failure_prefix_saved"] is True
    assert row["failure_vector_prefix_available"] is False
    assert "models_artifact" not in row


def test_artifact_reservation_checks_before_writing(tmp_path):
    (tmp_path / "existing").write_bytes(b"1234")
    with pytest.raises(OSError, match="envelope"):
        driver.checked_exclusive(tmp_path / "too_large", b"12345", limit=10, reserve=2)
    assert not (tmp_path / "too_large").exists()
    driver.checked_exclusive(tmp_path / "fits", b"1234", limit=10, reserve=2)
    assert driver.retained_bytes(tmp_path) == 8


def test_preparation_routes_old_raw_schema_and_registers_completed_observation_work():
    p, old = protocol()
    preparation = driver.preparation_protocol(old, p)
    initial, truth = np.full(2401, .25), np.full(2401, .2)
    raw = dict(raw_model=np.full((24, 24), .2), identity=dict(slice_id="manufactured"))
    calls, providers = [], {}

    def load(_, received):
        assert received == old
        calls.append("old_schema_load")
        return [raw]

    def prepare(received, received_protocol):
        assert received is raw and received_protocol["geometry"]["frequencies_hz"] == driver.FREQUENCIES
        calls.append("all8_data")
        return dict(initial=initial, truth=truth, blocks=(), receivers=(1, 2),
            truth_sha256="manufactured_truth", raw_model_sha256="manufactured_raw", preparation_costs=dict(forward_rhs=48))

    runtime = SimpleNamespace(np=np, helpers=SimpleNamespace(load_raw_inputs=load, prepare_case=prepare),
        controller=SimpleNamespace(model_sha=lambda a: hashlib.sha256(a.tobytes()).hexdigest(),
            PreparedProblem=lambda *args: args))
    cost_context = SimpleNamespace(register_preparation_costs=lambda name, provider: providers.update({name: provider}))
    context = {}
    result = driver.prepare_for_run(REPO, old, preparation, dict(case_index=0), runtime,
        context, None, lambda: calls.append("static_check"), cost_context)
    assert calls == ["static_check", "old_schema_load", "all8_data"]
    assert isinstance(result[0], dict) and "truth" not in result[0]
    assert providers["all8_observation_generation"]() == dict(forward_rhs=48)
    assert context["common_binding"]["initial_native_m_sha256"] == runtime.controller.model_sha(initial)


def test_real_manufactured_49_grid_bootstrap_and_independent_common8_audit(tmp_path, monkeypatch):
    """The actual imported807 and controller APIs; no corpus NPZ is opened."""
    p, old = protocol()
    for name in ["controller.py", "bounded_exact.py"]:
        shutil.copyfile(SOURCE_CONTROLLER / name, tmp_path / name)
        p["local_dependency_sha256"][name] = driver.sha(tmp_path / name)
    shutil.copyfile(HERE / "independent_saved_models.py", tmp_path / "independent_saved_models.py")
    runtime = driver.load_runtime(REPO, tmp_path, p)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("manufactured integration reads no NPZ/corpus"))
    zz, xx = np.indices((24, 24), dtype=np.float64)
    velocity = 2.5 + .3*np.sin(.21*zz)*np.cos(.18*xx) + .2*(zz > 11)
    raw = dict(raw_model=1/(velocity*velocity), identity=dict(slice_id="manufactured_only"))
    preparation, context = driver.preparation_protocol(old, p), dict(raw_case=raw)

    def prepare(cost_context):
        prepared = runtime.helpers.prepare_case(raw, preparation)
        context["prepared"] = prepared
        counters = copy.deepcopy(prepared["preparation_costs"])
        cost_context.register_preparation_costs("all8_observation_generation", lambda: counters)
        return runtime.controller.PreparedProblem(dict(preparation["physics"]), prepared["initial"],
            prepared["blocks"], prepared["receivers"])

    settings = runtime.controller.ControllerSettings(engine_deadline_seconds=20., first_stage_deadline_seconds=10.,
        api_return_cap_seconds=30., single_stage_maximum_steps=0, per_stage_maximum_steps=0)
    measured = runtime.controller.measure_controller_return(arm="joint8", prepare=prepare,
        api=runtime.api, settings=settings)
    payload = measured["payload"]
    assert measured["runtime_gate"]["eligible"] is True
    assert payload["common_initial_full8"]["objective"] > 1e-6
    assert payload["common_full8_ratio"] == 1.
    assert payload["preparation_costs"]["all8_observation_generation"]["latest_counts"]["forward_rhs"] == 48
    audit = driver.offline_audit(dict(case_index=0), context, payload, p, old, runtime)
    assert not audit["failures"] and max(audit["observation_relative_L2"].values()) <= 1e-9
    assert audit["records"]["initial"]["relative_physical_velocity_L2"] > 0
    assert driver.scalar_trace(payload)["checkpoint_vectors_retained"] is False
    print(json.dumps(dict(check="manufactured_common8_import_integration", cases=1, project_inputs_opened=0,
        all8_observation_RHS=48, numerical_checks=audit["checks"], failures=len(audit["failures"]),
        maximum_observation_relative_L2=max(audit["observation_relative_L2"].values()),
        scope="manufactured nonconstant49-grid API control, zero optimizer steps; no project result")))
