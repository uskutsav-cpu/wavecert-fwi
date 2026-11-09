"""Pure reader/phase-prefix controls; synthetic files, never BP inputs."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_numpy_bp_phase_tests", HERE/"run_replay.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_default_cli_never_executes_project_path(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(HERE/"run_replay.py")])
    monkeypatch.setattr(driver, "execute", lambda *args: pytest.fail("execution forbidden"))
    assert driver.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "metadata_only_no_BP_or_propagation"
    assert report["own_solves"] == 3 and report["direction_checks"] == 6


def phase_fixture(monkeypatch, tmp_path):
    original_spec = importlib.util.spec_from_file_location
    attempts = []

    class FakeCoreLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.physical_directions = lambda m: np.zeros((2, *m.shape))

            def fail(*args, **kwargs):
                attempts.append("synthetic_own_solve_attempt")
                raise RuntimeError("manufactured own-solve failure")

            module.propagate = fail

    def fake_spec(name, path):
        if name == "_numpy_bp_tangent_pinned_core":
            return importlib.util.spec_from_loader(name, FakeCoreLoader())
        return original_spec(name, path)

    monkeypatch.setattr(driver.importlib.util, "spec_from_file_location", fake_spec)
    monkeypatch.setattr(driver, "read_npz", lambda *args: pytest.fail("native numerical decode forbidden"))
    inputs, native = tmp_path/"synthetic_inputs", tmp_path/"synthetic_native"
    inputs.mkdir()
    native.mkdir()
    # Deliberately invalid JSON: its hash may be checked, never decoded here.
    (native/"complete_receipt.json").write_bytes(b"synthetic terminal hash marker, NOT JSON")
    terminal = tmp_path/"synthetic_terminal.json"
    terminal.write_text('{"manufactured_only":true}\n')
    p = json.loads((HERE/"protocol.json").read_text())
    p["inputs"] = []
    for index in range(3):
        path = inputs/f"synthetic_{index}.npy"
        np.save(path, np.full((96,192), 2000.+index, dtype=np.float32))
        p["inputs"].append(dict(id=f"synthetic_{index}", file=path.name, sha256=driver.sha(path)))
    args = SimpleNamespace(protocol=HERE/"protocol.json", inputs=inputs, native_run=native,
        output=tmp_path/"audit_output", terminal_observation=terminal,
        expected_native_complete_sha256=driver.sha(native/"complete_receipt.json"),
        expected_source_sha256=driver.sha(HERE/"run_replay.py"),
        expected_protocol_sha256=driver.sha(HERE/"protocol.json"))
    return args,p,attempts


def test_all_three_failed_own_solves_keep_six_denominators_and_native_unopened(monkeypatch, tmp_path):
    args,p,attempts = phase_fixture(monkeypatch,tmp_path)
    with pytest.raises(RuntimeError,match="own solves incomplete"):
        driver.execute(args,p)
    status = json.loads((args.output/"status.json").read_text())
    assert len(attempts) == len(status["cases"]) == 3
    assert sum(len(row["directions"]) for row in status["cases"]) == 6
    assert all(row["status"] == "failed_own_solve" for row in status["cases"])
    assert status["native_numeric_arrays_opened"] is False
    assert status["numerical_agreement_pass"] is False
    assert not (args.output/"complete_receipt.json").exists()


def test_wrong_native_completion_hash_blocks_inputs_and_all_own_solves(monkeypatch, tmp_path):
    args,p,attempts = phase_fixture(monkeypatch,tmp_path)
    args.expected_native_complete_sha256 = "0"*64
    monkeypatch.setattr(np,"load",lambda *a,**k: pytest.fail("no input arrays before terminal gate"))
    with pytest.raises(ValueError,match="terminal gate"):
        driver.execute(args,p)
    status = json.loads((args.output/"status.json").read_text())
    assert not attempts and all(row["status"] == "unattempted" for row in status["cases"])
    assert status["native_numeric_arrays_opened"] is False
    assert len(status["cases"]) == 3 and sum(len(r["directions"]) for r in status["cases"]) == 6
