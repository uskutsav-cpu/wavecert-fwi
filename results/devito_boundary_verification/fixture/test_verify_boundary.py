"""Tiny synthetic metric/layout/guard checks; no Devito propagation or project data."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import verify_boundary as module

PROTOCOL = Path(__file__).with_name("protocol.json")


def test_fixed_seven_cases_have_same_exterior_control_grids_and_node_locations():
    protocol, _ = module.load_protocol(PROTOCOL)
    cases = {case["id"]: case for case in protocol["cases"]}

    def record(case):
        positions = module.geometry(case, protocol["common"])
        h = np.asarray(protocol["common"]["spacing_m"])
        origin = -case["nbl"]*h
        grid = (np.asarray(case["shape_x_z"])+2*case["nbl"]).tolist()
        return dict(physics_before=dict(vp={"constant": 2., "shape": grid},
            derived_squared_slowness={"constant": .25, "shape": grid},
            grid_shape=grid, grid_spacing=h.tolist()),
            computational_source_index=((np.asarray([positions["source_x_z_m"]])-origin)/h).tolist(),
            computational_receiver_index=((np.asarray([positions["receiver_x_z_m"]])-origin)/h).tolist())

    for damped, undamped, expected_source in [("small_nbl20", "matched_exterior_20_undamped", 50.),
                                             ("small_nbl40", "matched_exterior_40_undamped", 70.)]:
        first, second = record(cases[damped]), record(cases[undamped])
        assert all(module.matched_layout(first, second).values())
        assert first["computational_source_index"] == [[expected_source, expected_source]]
        assert first["computational_receiver_index"] == [[expected_source+5., expected_source]]
        second["physics_before"]["grid_spacing"][0] = 20.
        assert not all(module.matched_layout(first, second).values())
    assert max((c["shape_x_z"][0]+2*c["nbl"]) for c in cases.values()) == 401
    assert 3*401**2*8 == protocol["memory"]["maximum_nominal_three_slice_bytes"]


def test_nominal_image_path_estimates_are_labeled_and_do_not_claim_discrete_causality():
    protocol, _ = module.load_protocol(PROTOCOL)
    cases = {case["id"]: case for case in protocol["cases"]}
    for name, expected in [("small_nbl0", 275.), ("reference_241", 1175.), ("reference_321", 1575.)]:
        result = module.geometry(cases[name], protocol["common"])
        assert result["nearest_physical_image_travel_ms"] == expected
        assert "no strict discrete" in result["scope"]


def test_metrics_have_exact_window_boundary_and_explicit_reference_normalizers():
    reference = np.array([[1.], [1.], [2.], [2.]])
    candidate = reference+np.array([[1.], [-1.], [3.], [-4.]])
    result = module.trace_metrics(candidate, reference, np.array([0., 250., 251., 1100.]))
    assert result["early_samples"] == result["late_samples"] == 2
    assert result["early_relative"]["value"] == pytest.approx(1.)
    assert result["late_mismatch_over_full_reference"]["value"] == pytest.approx(5/np.sqrt(10))
    assert result["full_relative"]["value"] == pytest.approx(np.sqrt(27/10))
    assert result["quality_threshold"] is None and not result["physics_pass_claim"]


def test_zero_reference_norm_is_explicit_null_not_a_success_value():
    result = module.trace_metrics(np.ones((4, 1)), np.zeros((4, 1)), np.array([0., 250., 251., 1100.]))
    for name in ["early_relative", "late_mismatch_over_full_reference", "full_relative"]:
        assert result[name]["value"] is None and result[name]["denominator_zero"]
        assert result[name]["numerator"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("invalid", ["wrong_shape", "nan", "nonincreasing_times"])
def test_metrics_reject_invalid_layouts_instead_of_forming_partial_ratios(invalid):
    reference, candidate = np.ones((4, 1)), np.ones((4, 1))
    times = np.array([0., 250., 251., 1100.])
    if invalid == "wrong_shape":
        candidate = candidate[:-1]
    elif invalid == "nan":
        candidate[0] = np.nan
    else:
        times[2] = times[1]
    with pytest.raises(ValueError):
        module.trace_metrics(candidate, reference, times)


def test_default_cli_and_wrong_source_hash_never_call_PDE_or_create_output(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(module, "source_receipt", lambda *_: pytest.fail("PDE package import was reached"))
    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "protocol_only_no_PDE"
    with pytest.raises(ValueError, match="hash mismatch"):
        module.run(PROTOCOL, tmp_path, tmp_path/"forbidden", "wrong", "wrong")
    assert not (tmp_path/"forbidden").exists()


def test_existing_empty_output_directory_is_preserved_before_backend_import(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "source_receipt", lambda *_: pytest.fail("PDE package import was reached"))
    output = tmp_path/"existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.run(PROTOCOL, tmp_path, output, module.sha(PROTOCOL.read_bytes()), module.sha(Path(module.__file__).read_bytes()))
    assert output.is_dir() and list(output.iterdir()) == []


def test_changed_fixture_layout_is_rejected_before_any_execution(tmp_path):
    protocol, _ = module.load_protocol(PROTOCOL)
    protocol = copy.deepcopy(protocol)
    protocol["cases"][3]["translation_m"][0] += 10.
    changed = tmp_path/"protocol.json"
    changed.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="seven-case"):
        module.load_protocol(changed)


def test_interrupted_attempt_is_persisted_before_mock_propagation_and_remains_in_failed_receipt(monkeypatch, tmp_path):
    output = tmp_path/"interrupted_run"
    monkeypatch.setattr(module, "source_receipt", lambda *_: {"scope": "mock source guard; no PDE"})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=200*1024**2))

    def interrupted(case, common):
        before = json.loads((output/"status.json").read_text())
        assert before["status"] == "running"
        assert len(before["case_records"]) == 1
        assert before["case_records"][0]["case_id"] == case["id"] == "small_nbl0"
        assert before["case_records"][0]["status"] == "attempted"
        raise KeyboardInterrupt("synthetic interruption; no PDE")

    monkeypatch.setattr(module, "execute_case", interrupted)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        module.run(PROTOCOL, tmp_path, output, module.sha(PROTOCOL.read_bytes()), module.sha(Path(module.__file__).read_bytes()))
    after = json.loads((output/"status.json").read_text())
    assert after["status"] == "failed" and len(after["case_records"]) == 1
    attempt = after["case_records"][0]
    assert attempt["status"] == "interrupted" and attempt["finished_utc"]
    assert "KeyboardInterrupt" in attempt["error"]
    assert not (output/"complete_receipt.json").exists()
    assert not list(output.glob("*_raw.npz"))
