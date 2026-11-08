"""Inert engineering helper checks; no project models, merit solves or clocks."""

import gzip
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import adapter, verification


def test_candidate_config_is_explicitly_ai_authored_unapproved_and_not_student_final_design(tmp_path):
    path = Path(__file__).with_name("protocol.json")
    p, raw = verification._read_protocol(path)
    assert "AI-authored" in p["role"] and "SOFTWARE" in p["role"]
    assert not p["student_final_research_design"] and not p["competition_evidence"]
    assert not p["project_execution_authorized"] and p["candidate_controls_and_defaults_only"]
    assert "No project-study runner" in p["execution_gate"]
    assert not hasattr(verification, "execute") and not hasattr(verification, "main")
    changed = tmp_path/"candidate.json"
    changed.write_bytes(raw.replace(b'"rank": 64', b'"rank": 32'))
    with pytest.raises(ValueError, match="changed"):
        verification._read_protocol(changed)


def test_provisional_schedule_is_deterministic_balanced_and_retains_every_control():
    p, _ = verification._read_protocol(Path(__file__).with_name("protocol.json"))
    schedule = verification.scheduled_runs(p)
    assert schedule == verification.scheduled_runs(p) and len(schedule) == 120
    for repetition in range(3):
        for case in range(8):
            rows = [r for r in schedule if r["repetition"] == repetition and r["case_index"] == case]
            assert len(rows) == 5 and {r["method"] for r in rows} == set(adapter.METHODS)
            assert sorted(r["order_position"] for r in rows) == list(range(5))


def test_refinement_preserves_coordinates_and_constant_velocity_without_inspecting_project_arrays():
    velocity = verification.refined_velocity(np.full((24, 24), .25), 49)
    assert velocity.shape == (49, 49) and np.allclose(velocity, 2.)
    for index in (0, 50, 54, 58, 61, 65, 69, 575):
        mapped = verification.refined_index(index)
        old_z, old_x = divmod(index, 24)
        new_z, new_x = divmod(mapped, 49)
        assert (old_z+1)*.05 == pytest.approx((new_z+1)*.025)
        assert (old_x+1)*.05 == pytest.approx((new_x+1)*.025)
    with pytest.raises(ValueError):
        verification.refined_index(True)


def _trajectory():
    models = [np.array([.2, .3]), np.array([.21, .31]), np.array([.22, .32])]
    checkpoints = [dict(model=m, model_sha256=adapter.array_sha(m), objective=phi,
        elapsed_seconds=t, within_budget=True, online_costs={"fine_rhs": i+1})
        for i, (m, phi, t) in enumerate(zip(models, (4., 1.5, .7), (.6, 1.2, 4.1), strict=True))]
    return dict(initial_objective=4., final_model=models[-1], checkpoints=checkpoints,
        steps=[dict(model_sha256=checkpoints[0]["model_sha256"], proposal={"direction": np.array([-.1, -.2])},
            trials=[dict(objective=5., accepted=False, elapsed_seconds=.9)], costs={"fine_rhs": 2})]), models


def test_budget_ladder_credits_only_completed_prefix_and_unreached_targets_are_censored():
    result, models = _trajectory()
    protocol = dict(endpoint_budgets_seconds=[.5, 1., 2., 4., 8.], reported_objective_ratios=[.5, .25, .1])
    endpoints, targets = verification.budget_endpoints(result, {"initial": models[0]}, protocol)
    assert not endpoints[0]["completed_paid_checkpoint"] and endpoints[0]["online_objective"] is None
    assert [e["online_objective"] for e in endpoints[1:]] == [4., 1.5, 1.5, .7]
    assert [r["elapsed_seconds"] for r in targets] == [1.2, 4.1, None]
    assert not targets[-1]["reached"]
    assert np.array_equal(endpoints[-1]["model"], models[-1])


def test_compact_receipts_keep_all_events_hashes_and_budget_models_without_dense_json_duplicates(tmp_path):
    result, models = _trajectory()
    endpoints, targets = verification.budget_endpoints(result, {"initial": models[0]},
        dict(endpoint_budgets_seconds=[1., 2., 8.], reported_objective_ratios=[.5]))
    verification._save_run(tmp_path, 0, result, endpoints, targets, dict(costs={"fine_rhs": 11}, policy_access=False))
    with gzip.open(tmp_path/"run_000.json.gz", "rt") as stream:
        receipt = json.load(stream)
    assert len(receipt["trajectory"]["checkpoints"]) == len(result["checkpoints"])
    assert receipt["trajectory"]["checkpoints"][1]["model"]["sha256"] == adapter.array_sha(models[1])
    assert not receipt["trajectory"]["checkpoints"][1]["model"]["dense_values_saved"]
    assert receipt["trajectory"]["steps"][0]["trials"][0]["objective"] == 5.
    with np.load(tmp_path/receipt["models_artifact"], allow_pickle=False) as archive:
        assert set(archive.files) == {"initial", "budget_0", "budget_1", "budget_2", "final"}
        assert np.array_equal(archive["budget_1"], models[1])
    with pytest.raises(ValueError, match="overwriting"):
        verification._save_run(tmp_path, 0, result, endpoints, targets, {})


def test_compact_array_receipts_preserve_complex_bytes_and_dtype_without_lossy_cast():
    first = verification.compact_trajectory(np.array([1.+2j], dtype=">c16"))
    second = verification.compact_trajectory(np.array([1.+3j], dtype="<c16"))
    assert first["array_dtype"] == second["array_dtype"] == "<c16"
    assert first["sha256"] != second["sha256"]
    assert first == verification.compact_trajectory(np.array([1.+2j], dtype="<c16"))
    with pytest.raises(ValueError, match="object arrays"):
        verification.compact_trajectory(np.array([{}], dtype=object))


def test_module_binding_checks_actual_loaded_checkout_not_just_sibling_source_bytes(tmp_path, monkeypatch):
    import wavecert

    root = Path(wavecert.__file__).resolve().parents[2]
    receipt = verification.module_receipt(root)
    assert set(receipt) == set(verification.MODULES)
    module = importlib.import_module("wavecert.physics.exact_merit")
    foreign = tmp_path/"exact_merit.py"
    foreign.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(foreign))
    with pytest.raises(RuntimeError, match="foreign cached"):
        verification.module_receipt(root)
