"""Manufactured arrays and stub propagation only: never run a project PDE."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ci = module("_canonical_under_test", BASE / "canonical_inputs.py")
runner = module("_runtime_under_test", BASE / "run_replay.py")


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def manufactured(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    rows, calls = [], []
    for index, name in enumerate(["upper_left", "middle_center", "lower_right"]):
        folder = native / name
        folder.mkdir()
        vp = np.full(ci.SHAPE, 2.0 + index, dtype="<f8")
        mass = 1 / (vp * vp)
        directions = [np.full(ci.SHAPE, .01 + index, dtype="<f8"), np.full(ci.SHAPE, .02 + index, dtype="<f8")]
        # Poisoned target members must remain unread by exporter.
        np.savez(folder / "probe_saved_record.npz", actual_physical_velocity_kmps=vp,
                 receiver=np.asarray([object()], dtype=object))
        witnesses = {}
        for label, direction in zip(["interior", "edge"], directions, strict=True):
            path = folder / (label + "_direction.npz")
            np.savez(path, probe_m=mass, direction=direction, residual=np.asarray([object()], dtype=object))
            witnesses[label] = dict(artifact=path.name, artifact_sha256=ci.sha(path),
                probe_m_sha256=ci.array_sha(mass, np), direction_sha256=ci.array_sha(direction, np), norm_scale=1.0)
        case = dict(id=name, file=name + ".npy", sha256="raw-ancestry-" + name)
        call = dict(case_id=name, call_id="probe_saved", result=dict(artifact="probe_saved_record.npz"))
        status = dict(case=case, calls=[call], direction_witnesses=witnesses)
        dump(folder / "status.json", status)
        dump(folder / "complete_receipt.json", status)
        rows.append(dict(case=case, worker_status_sha256=ci.sha(folder / "status.json"),
            worker_complete_sha256=ci.sha(folder / "complete_receipt.json"),
            probe_call_canonical_sha256=ci.canonical_sha(call), probe_artifact="probe_saved_record.npz",
            probe_artifact_sha256=ci.sha(folder / "probe_saved_record.npz"),
            physical_velocity_sha256=ci.array_sha(vp, np), actual_physical_m_sha256=ci.array_sha(mass, np),
            witnesses=witnesses))
        calls.append(call)
    dump(native / "complete_receipt.json", dict(calls=calls))
    plan = dict(cases=rows, native_complete_sha256=ci.sha(native / "complete_receipt.json"))
    path = tmp_path / "plan.json"
    dump(path, plan)
    return native, path, plan


def exported(tmp_path):
    native, path, plan = manufactured(tmp_path)
    output = tmp_path / "inputs"
    ci.export(path, native, output, ci.sha(path), ci.sha(Path(ci.__file__)))
    manifest = ci.load_manifest(output, ci.sha(output / "manifest.json"), plan,
                                ci.sha(path), ci.sha(Path(ci.__file__)))
    return native, path, plan, output, manifest


def test_exact_export_ignores_poison_target_members(tmp_path):
    _, _, plan, output, manifest = exported(tmp_path)
    for row, binding in zip(manifest["cases"], plan["cases"], strict=True):
        arrays, _ = ci.load_case(output, manifest, row["case"], np)
        assert set(arrays) == set(ci.KEYS)
        assert all(a.flags.c_contiguous for a in arrays.values())
        assert {key: ci.array_sha(a, np) for key, a in arrays.items()} == ci.expected_arrays(binding)
    complete = json.loads((output / "export_complete.json").read_text())
    assert complete["PDE_calls"] == complete["native_solution_outputs_decoded"] == 0


def test_member_allowlist_denies_target_access(tmp_path):
    path = tmp_path / "poison.npz"
    np.savez(path, receiver=np.ones(2))
    with pytest.raises(ValueError, match="forbidden"):
        ci.selected_member(path, "receiver", {"direction", "probe_m"}, np)


@pytest.mark.parametrize("kind", ["source", "plan", "worker", "probe", "witness"])
def test_export_provenance_corruption_fails(tmp_path, kind):
    native, path, plan = manufactured(tmp_path)
    source_hash, plan_hash = ci.sha(Path(ci.__file__)), ci.sha(path)
    if kind == "source":
        source_hash = "0" * 64
    elif kind == "plan":
        plan_hash = "0" * 64
    else:
        name = {"worker": "status.json", "probe": "probe_saved_record.npz", "witness": "interior_direction.npz"}[kind]
        with (native / "upper_left" / name).open("ab") as stream:
            stream.write(b"corruption")
    with pytest.raises(ValueError):
        ci.export(path, native, tmp_path / "bad", plan_hash, source_hash)


@pytest.mark.parametrize("kind", ["manifest_sha", "wrong_case", "wrong_binding", "array_sha", "container_sha", "missing", "extra", "duplicate", "nan", "zero_direction", "wrong_dtype", "negative_mass"])
def test_canonical_corruption_fails(tmp_path, kind):
    _, path, plan, output, manifest = exported(tmp_path)
    row = manifest["cases"][0]
    case = row["case"]
    if kind == "manifest_sha":
        with pytest.raises(ValueError):
            ci.load_manifest(output, "0" * 64, plan, ci.sha(path), ci.sha(Path(ci.__file__)))
        return
    if kind == "wrong_binding":
        manifest["cases"][0]["native_binding"]["worker_status_sha256"] = "bad"
        dump(output / "manifest.json", manifest)
        with pytest.raises(ValueError):
            ci.load_manifest(output, ci.sha(output / "manifest.json"), plan, ci.sha(path), ci.sha(Path(ci.__file__)))
        return
    if kind == "wrong_case":
        case = dict(case, id="unregistered")
    elif kind == "container_sha":
        row["artifact_sha256"] = "0" * 64
    elif kind == "array_sha":
        row["array_sha256"]["edge_direction"] = "0" * 64
    else:
        file = output / row["artifact"]
        with np.load(file) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if kind == "missing":
            arrays.pop("edge_direction")
        elif kind == "extra":
            arrays["receiver"] = np.ones(2)
        elif kind == "nan":
            arrays["edge_direction"][0, 0] = np.nan
        elif kind == "zero_direction":
            arrays["edge_direction"][:] = 0
        elif kind == "wrong_dtype":
            arrays["edge_direction"] = arrays["edge_direction"].astype(np.float32)
        elif kind == "negative_mass":
            arrays["direction_parameter_probe_m"][0, 0] = -1
        np.savez(file, **arrays)
        if kind == "duplicate":
            import zipfile
            with zipfile.ZipFile(file, "a") as archive:
                archive.writestr("edge_direction.npy", b"duplicate")
        row["artifact_sha256"] = ci.sha(file)
    with pytest.raises(ValueError):
        ci.load_case(output, manifest, case, np)


def test_frozen_core_and_criteria_unchanged():
    p, _ = runner.load_protocol(BASE / "protocol.json")
    old = json.loads((BASE / "history/v1_c51_f0/protocol.json").read_text())
    assert p["criteria"] == old["criteria"]
    assert ci.sha(BASE / "numpy_bp_tangent.py") == old["core_sha256"]


def test_no_direction_regeneration_in_v2_runtime():
    text = (BASE / "run_replay.py").read_text()
    assert "core.physical_directions(" not in text
    assert "model=1/(vp*vp)" not in text


@pytest.mark.parametrize("fail_one_solve", [False, True])
def test_stub_all_three_solves_before_target_output_comparisons(tmp_path, monkeypatch, fail_one_solve):
    _, _, plan, inputs, manifest = exported(tmp_path)
    p = json.loads((BASE / "protocol.json").read_text())
    native_p = json.loads((BASE / "original_native_protocol.json").read_text())
    cases = [r["case"] for r in plan["cases"]]
    p["inputs"] = cases
    native_p["inputs"] = cases
    calls = []
    for case in cases:
        prefix = case["id"]
        calls.extend(dict(case_id=prefix, call_id=v["id"], kind="forward", status="complete") for v in native_p["forward_calls"])
        calls.extend([dict(case_id=prefix, call_id="probe_saved", kind="forward_saved", status="complete"),
                      dict(case_id=prefix, call_id="probe_adjoint", kind="adjoint", status="complete")])
        calls.extend(dict(case_id=prefix, call_id=f"fd_{d}_{e}_{sign}", kind="forward_fd", status="complete")
                     for d in p["directions"] for e in native_p["gradient"]["relative_steps"] for sign in ["plus", "minus"])
    native = tmp_path / "runtime_native"
    native.mkdir()
    dump(native / "complete_receipt.json", dict(status="complete_guard_attempts", source_sha256=p["native_producer_sha256"],
        protocol_sha256=p["original_native_protocol_sha256"], physical_acceptance=None, calls=calls))
    # Synthetic runtime plan is separately frozen; this is not actual native evidence.
    plan["native_complete_sha256"] = ci.sha(native / "complete_receipt.json")
    manifest["native_complete_sha256"] = plan["native_complete_sha256"]
    fixture = tmp_path / "runtime_fixture"
    fixture.mkdir()
    dump(fixture / "EXPORT_SOURCE_PLAN.json", plan)
    dump(fixture / "original_native_protocol.json", native_p)
    (fixture / "canonical_inputs.py").write_bytes((BASE / "canonical_inputs.py").read_bytes())
    p["export_source_plan_sha256"] = ci.sha(fixture / "EXPORT_SOURCE_PLAN.json")
    manifest["plan_sha256"] = p["export_source_plan_sha256"]
    dump(inputs / "manifest.json", manifest)
    dump(fixture / "protocol.json", p)
    events = []

    class StubLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, target):
            def propagate(vp, directions, **params):
                assert len([e for e in events if e == "compare"]) == 0
                events.append("solve")
                if fail_one_solve and events.count("solve") == 1:
                    raise ValueError("manufactured own-solve failure")
                nt = params["nt"]
                return dict(receiver=np.zeros((nt, 10)), tangent_receiver=np.zeros((2, nt, 10)),
                    times_ms=np.arange(nt) * .5, computed_mask=np.ones(nt, dtype=bool), source=np.zeros((nt, 1))), dict(native_numeric_witnesses_used=False)
            target.propagate = propagate

    original_spec = importlib.util.spec_from_file_location

    def choose_spec(name, path):
        if Path(path).name == "numpy_bp_tangent.py":
            return importlib.machinery.ModuleSpec(name, StubLoader())
        return original_spec(name, path)

    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location", choose_spec)
    monkeypatch.setattr(runner, "load_protocol", lambda path: (p, native_p))

    def compare(case, own, own_arrays, native_run, protocol, native_protocol, expected_calls, np_, opening, opened, trace, direction):
        assert events.count("solve") == 3
        assert own["native_derived_numeric_inputs_used"] is True
        assert own["native_solution_outputs_used_during_own_solve"] is False
        assert own["kernel_opens_native_files"] is False
        assert "native_numeric_witnesses_used" not in own
        events.append("compare")
        opening()
        opened()
        return dict(trace_agreement=True, derivatives=[dict(direction=d, agreement=True) for d in p["directions"]])

    monkeypatch.setattr(runner, "compare_case", compare)
    observation = tmp_path / "terminal.json"
    dump(observation, dict(role="manufactured terminal marker"))
    args = SimpleNamespace(protocol=fixture / "protocol.json", output=tmp_path / "replay", inputs=inputs, native_run=native,
        terminal_observation=observation, expected_native_complete_sha256=ci.sha(native / "complete_receipt.json"),
        expected_source_sha256=ci.sha(Path(runner.__file__)), expected_protocol_sha256=ci.sha(fixture / "protocol.json"),
        expected_input_manifest_sha256=ci.sha(inputs / "manifest.json"))
    if fail_one_solve:
        with pytest.raises(RuntimeError, match="own solves incomplete"):
            runner.execute(args, p)
        assert events == ["solve"] * 3
        status = json.loads((args.output / "status.json").read_text())
        assert status["native_solution_output_arrays_opened"] is False
        assert not (args.output / "complete_receipt.json").exists()
        assert len(status["cases"]) == 3
        return
    assert runner.execute(args, p) == 0
    assert events == ["solve"] * 3 + ["compare"] * 3
    status = json.loads((args.output / "complete_receipt.json").read_text())
    assert status["native_derived_numeric_inputs_used"] is True
    assert status["native_solution_outputs_used_during_own_solves"] is False
    assert status["full3_case_6_direction_denominators_retained"] is True
