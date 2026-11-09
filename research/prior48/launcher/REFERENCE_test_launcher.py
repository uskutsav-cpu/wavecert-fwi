"""Only tiny dummy subprocesses; no project imports, arrays, PDEs or timing study."""
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).with_name("launch_cohort.py").resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_json(path):
    until = time.monotonic() + 8
    while time.monotonic() < until:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(.02)
    raise AssertionError("dummy process receipt not written: " + str(path))


def fixture(tmp_path, *, code=0, seconds=.05):
    repo = tmp_path / "repo"
    repo.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    driver = bundle / "dummy_driver.py"
    driver.write_text("import argparse,time,sys\n"
        "p=argparse.ArgumentParser();p.add_argument('--output');a,_=p.parse_known_args()\n"
        "print('DUMMY NO PROJECT WORK',flush=True)\n"
        f"time.sleep({seconds!r})\nsys.exit({code!r})\n")
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(json.dumps(dict(planned=96, complete=85, attempted=1,
        unattempted=10, actual_exit_code=None)))
    protocol = bundle / "protocol.json"
    protocol.write_text(json.dumps(dict(threads={"OMP_NUM_THREADS":"1"}, cohort_context=dict(
        cohort_id="manufactured_launcher_control", predecessor_external_observation_sha256=digest(predecessor)))))
    value = dict(schema_version=1, cohort_id="manufactured_launcher_control", schedule=dict(
        cases=8, methods=4, repetitions=3, total_runs=96), python=str(Path(sys.executable).resolve()),
        driver=str(driver), protocol=str(protocol), repo=str(repo), output=str(tmp_path / "new_output"),
        predecessor_output=str(tmp_path / "old_output"), predecessor_receipt=str(predecessor),
        expected_driver_sha256=digest(driver), expected_protocol_sha256=digest(protocol),
        expected_predecessor_receipt_sha256=digest(predecessor), threads={"OMP_NUM_THREADS":"1"})
    manifest = tmp_path / "registration.json"
    manifest.write_text(json.dumps(value))
    job = tmp_path / "supervision"
    command = [sys.executable, str(LAUNCHER), "--launch", "--manifest", str(manifest),
        "--job-directory", str(job), "--expected-launcher-sha256", digest(LAUNCHER),
        "--expected-manifest-sha256", digest(manifest)]
    return value, manifest, job, command


@pytest.mark.parametrize("code", [0, 7])
def test_actual_exit_pid_and_no_false_completion(tmp_path, code):
    _, manifest, job, command = fixture(tmp_path, code=code)
    original = manifest.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0
    dispatch = json.loads(result.stdout)
    started = wait_json(job / "supervisor_started.json")
    child = wait_json(job / "child_started.json")
    end = wait_json(job / "child_exit.json")
    wait_json(job / "supervisor_exit.json")
    assert started["PID"] == dispatch["supervisor_PID"] == child["supervisor_PID"]
    assert started["session_ID"] == started["PID"]
    assert child["PID"] == end["child_PID"] and child["PID"] != started["PID"]
    assert child["command"][1:4] == ["-B", "-I", "-u"]
    assert end["actual_returncode"] == code and end["signal_if_negative"] is None
    assert end["complete_receipt_exists"] is False
    assert end["process_exit_is_not_independent_study_integrity_validation"] is True
    assert (job / "registration.json").read_bytes() == original
    assert "DUMMY NO PROJECT WORK" in (job / "producer.log").read_text()


def test_dispatch_returns_while_dummy_keeps_running(tmp_path):
    _, _, job, command = fixture(tmp_path, seconds=.6)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0
    wait_json(job / "child_started.json")
    assert not (job / "child_exit.json").exists()
    assert wait_json(job / "child_exit.json")["actual_returncode"] == 0


def test_supervisor_signal_forwarding_records_real_signal_exit(tmp_path):
    _, _, job, command = fixture(tmp_path, seconds=10)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0
    child = wait_json(job / "child_started.json")
    os.kill(child["supervisor_PID"], signal.SIGTERM)
    end = wait_json(job / "child_exit.json")
    assert end["actual_returncode"] == -signal.SIGTERM
    assert end["signal_if_negative"] == signal.SIGTERM
    assert wait_json(job / "signal_001.json")["delivery"] == "forwarded_to_child_process_group"
    wait_json(job / "supervisor_exit.json")


@pytest.mark.parametrize("changed", ["manifest", "driver", "protocol", "predecessor"])
def test_hash_changes_prevent_dispatch(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    path = manifest if changed == "manifest" else Path(value[
        "predecessor_receipt" if changed == "predecessor" else changed])
    path.write_text(path.read_text() + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not job.exists()


def test_existing_output_and_job_are_never_overwritten(tmp_path):
    value, _, job, command = fixture(tmp_path)
    output = Path(value["output"])
    output.mkdir()
    witness = output / "old.txt"
    witness.write_text("preserve")
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and witness.read_text() == "preserve" and not job.exists()
    witness.unlink()
    output.rmdir()
    job.mkdir()
    saved = job / "old.txt"
    saved.write_text("preserve")
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and saved.read_text() == "preserve"


def test_default_cli_does_not_open_registration_or_launch(tmp_path):
    result = subprocess.run([sys.executable, str(LAUNCHER)], capture_output=True, text=True,
        timeout=5, check=False, cwd=tmp_path)
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "metadata_only_no_launch"
    assert list(tmp_path.iterdir()) == []


def test_dispatch_receipt_write_failure_tracks_and_reaps_known_worker(tmp_path, monkeypatch):
    _, manifest, job, _ = fixture(tmp_path, seconds=10)
    spec = importlib.util.spec_from_file_location("launcher_control_unique", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_record = module.record

    def fail_dispatch_receipt(path, value):
        if path.name == "dispatch_receipt.json":
            wait_json(job / "child_started.json")  # Exercise a genuinely live known child.
            raise OSError("manufactured post-spawn dispatch receipt failure")
        return real_record(path, value)

    monkeypatch.setattr(module, "record", fail_dispatch_receipt)
    with pytest.raises(OSError, match="manufactured"):
        module.dispatch(manifest, job, digest(manifest), digest(LAUNCHER))
    failure = json.loads((job / "dispatch_failure.json").read_text())
    cleanup = failure["cleanup"]
    assert cleanup["supervisor_PID"] is not None and cleanup["supervisor_returncode"] is not None
    with pytest.raises(ProcessLookupError):
        os.kill(cleanup["supervisor_PID"], 0)
    if cleanup["child_PID"] is not None and cleanup["actual_child_returncode"] is None:
        assert cleanup["in_flight_child_unknown"] is True


@pytest.mark.parametrize("changed", ["cohort_id", "protocol_predecessor"])
def test_registration_must_bind_protocol_cohort_and_predecessor(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    if changed == "cohort_id":
        value[changed] = "different"
    else:
        protocol = Path(value["protocol"])
        metadata = json.loads(protocol.read_text())
        metadata["cohort_context"]["predecessor_external_observation_sha256"] = "different"
        protocol.write_text(json.dumps(metadata))
        value["expected_protocol_sha256"] = digest(protocol)
    manifest.write_text(json.dumps(value))
    command[-1] = digest(manifest)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not job.exists()


@pytest.mark.parametrize("changed", ["output", "job"])
def test_predecessor_tree_cannot_receive_new_artifacts(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    previous = Path(value["predecessor_output"])
    previous.mkdir()
    target = previous / "forbidden_new_files"
    if changed == "output":
        value["output"] = str(target)
        manifest.write_text(json.dumps(value))
        command[-1] = digest(manifest)
    else:
        command[command.index("--job-directory") + 1] = str(target)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not target.exists() and not job.exists()
