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

LAUNCHER = Path(__file__).with_name("launch_prior48.py").resolve()


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
    driver.write_text("import argparse,time,sys,json,hashlib\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--root-go',required=True);a,_=p.parse_known_args()\n"
        "g=json.loads(Path(a.root_go).read_text());assert g['authorization']=='GO_CLASSICAL_PRIOR48_DEVELOPMENT'\n"
        "print('DUMMY NO PROJECT WORK; ROOT_GO_SEEN='+hashlib.sha256(Path(a.root_go).read_bytes()).hexdigest(),flush=True)\n"
        f"time.sleep({seconds!r})\nsys.exit({code!r})\n")
    previous = tmp_path / "protected_context"
    previous.mkdir()
    prior_receipt = tmp_path / "protected_context.json"
    prior_receipt.write_text(json.dumps(dict(role="MANUFACTURED prior-development context only", resume=False, pool=False)))
    identities = [{"index":i,"role":"MANUFACTURED metadata identity only"} for i in range(8)]
    methods = [{"name":"normalized_exact_data_only","lambda":0.0},
        {"name":"normalized_exact_fractional_update_prior","lambda":100.0}]
    rows = []
    for repetition in range(3):
        for case in range(8):
            for position, method in enumerate(methods):
                rows.append(dict(run_index=len(rows),case_index=case,repetition=repetition,
                    order_position=position,method=method["name"],weight=method["lambda"],
                    identity=identities[case],status="unattempted",final_study_eligible=False,primary_quality=None))
    schedule = bundle / "FROZEN48_SCHEDULE.json"
    schedule.write_text(json.dumps(rows))
    protocol = bundle / "protocol.json"
    production = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
    protocol.write_text(json.dumps(dict(threads={"OMP_NUM_THREADS":"1"},total_runs=48,
        identities=identities,methods=methods,production_source_sha256=production,
        production_file_count=161,root_go_authorization="GO_CLASSICAL_PRIOR48_DEVELOPMENT",
        schedule_sha256=digest(schedule))))
    go = bundle / "ROOT_GO_MANUFACTURED_ONLY.json"
    go.write_text(json.dumps(dict(authorization="GO_CLASSICAL_PRIOR48_DEVELOPMENT",
        driver_sha256=digest(driver),protocol_sha256=digest(protocol),schedule_sha256=digest(schedule),
        production_source_sha256=production,independent_source_review_clear=True,
        role="MANUFACTURED dummy-only GO; no project study authorization")))
    value = dict(schema_version=2,cohort_id="manufactured_prior48_launcher_control",
        schedule=dict(cases=8,methods=2,repetitions=3,total_runs=48),
        python=str(Path(sys.executable).resolve()),driver=str(driver),protocol=str(protocol),
        schedule_file=str(schedule),root_go=str(go),repo=str(repo),output=str(tmp_path/"new_output"),
        expected_driver_sha256=digest(driver),expected_protocol_sha256=digest(protocol),
        expected_schedule_file_sha256=digest(schedule),expected_root_go_sha256=digest(go),
        expected_production_source_sha256=production,expected_production_file_count=161,
        study_semantics="new_48_slot_development_no_resume_no_pooling",
        protected_prior_contexts=[dict(output=str(previous),receipt=str(prior_receipt),
            expected_receipt_sha256=digest(prior_receipt),no_resume=True,no_pooling=True,
            role="MANUFACTURED protected prior-development context only")],threads={"OMP_NUM_THREADS":"1"})
    manifest = tmp_path / "registration.json"
    manifest.write_text(json.dumps(value))
    job = tmp_path / "supervision"
    command = [sys.executable,str(LAUNCHER),"--launch","--manifest",str(manifest),
        "--job-directory",str(job),"--expected-launcher-sha256",digest(LAUNCHER),
        "--expected-manifest-sha256",digest(manifest)]
    return value,manifest,job,command


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
    assert "--root-go" in child["command"]
    assert "ROOT_GO_SEEN=" in (job / "producer.log").read_text()
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


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_supervisor_signal_forwarding_records_real_signal_exit(tmp_path,signum):
    _, _, job, command = fixture(tmp_path, seconds=10)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0
    child = wait_json(job / "child_started.json")
    os.kill(child["supervisor_PID"], signum)
    end = wait_json(job / "child_exit.json")
    assert end["actual_returncode"] == -signum
    assert end["signal_if_negative"] == signum
    assert wait_json(job / "signal_001.json")["delivery"] == "forwarded_to_child_process_group"
    wait_json(job / "supervisor_exit.json")


@pytest.mark.parametrize("changed", ["manifest", "driver", "protocol", "context", "schedule_file", "root_go", "launcher_pin"])
def test_hash_changes_prevent_dispatch(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    if changed == "launcher_pin":
        command[command.index("--expected-launcher-sha256")+1] = "0"*64
    else:
        path = manifest if changed == "manifest" else Path(value["protected_prior_contexts"][0]["receipt"] if changed == "context" else value[changed])
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
    metadata = json.loads(result.stdout)
    assert metadata["status"] == "metadata_only_no_launch"
    assert metadata["fixed_full_cohort"] == 48 and metadata["separate_root_go_required"] is True
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




@pytest.mark.parametrize("changed", ["output", "job"])
def test_protected_prior_tree_cannot_receive_new_artifacts(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    previous = Path(value["protected_prior_contexts"][0]["output"])
    target = previous / "forbidden_new_files"
    if changed == "output":
        value["output"] = str(target)
        manifest.write_text(json.dumps(value))
        command[-1] = digest(manifest)
    else:
        command[command.index("--job-directory") + 1] = str(target)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not target.exists() and not job.exists()


def rebind(value, manifest, command, *, schedule_changed=False, protocol_changed=False):
    if schedule_changed:
        value["expected_schedule_file_sha256"] = digest(Path(value["schedule_file"]))
        protocol = json.loads(Path(value["protocol"]).read_text())
        protocol["schedule_sha256"] = value["expected_schedule_file_sha256"]
        Path(value["protocol"]).write_text(json.dumps(protocol))
        protocol_changed = True
    if protocol_changed:
        value["expected_protocol_sha256"] = digest(Path(value["protocol"]))
        go = json.loads(Path(value["root_go"]).read_text())
        go["protocol_sha256"] = value["expected_protocol_sha256"]
        go["schedule_sha256"] = value["expected_schedule_file_sha256"]
        Path(value["root_go"]).write_text(json.dumps(go))
    value["expected_root_go_sha256"] = digest(Path(value["root_go"]))
    manifest.write_text(json.dumps(value))
    command[-1] = digest(manifest)


@pytest.mark.parametrize("field", ["authorization", "driver_sha256", "protocol_sha256",
    "schedule_sha256", "production_source_sha256", "independent_source_review_clear"])
def test_rehashed_go_content_must_match_exact_driver_contract(tmp_path, field):
    value, manifest, job, command = fixture(tmp_path)
    go = json.loads(Path(value["root_go"]).read_text())
    go[field] = 1 if field == "independent_source_review_clear" else "different"
    Path(value["root_go"]).write_text(json.dumps(go))
    rebind(value, manifest, command)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not job.exists()


@pytest.mark.parametrize("changed", ["missing_row", "duplicate_arm", "completed_status",
    "quality", "eligible", "identity", "duplicate_position", "wrong_index", "boolean_case", "boolean_weight"])
def test_rehashed_schedule_still_requires_complete_untouched48(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    path = Path(value["schedule_file"])
    rows = json.loads(path.read_text())
    if changed == "missing_row":
        rows.pop()
    elif changed == "duplicate_arm":
        rows[1]["method"], rows[1]["weight"] = rows[0]["method"], rows[0]["weight"]
    elif changed == "completed_status":
        rows[0]["status"] = "complete"
    elif changed == "quality":
        rows[0]["primary_quality"] = .1
    elif changed == "eligible":
        rows[0]["final_study_eligible"] = True
    elif changed == "identity":
        rows[0]["identity"] = {"wrong": "identity"}
    elif changed == "duplicate_position":
        rows[1]["order_position"] = 0
    elif changed == "wrong_index":
        rows[0]["run_index"] = 1
    elif changed == "boolean_case":
        rows[0]["case_index"] = False
    else:
        rows[0]["weight"] = False
    path.write_text(json.dumps(rows))
    rebind(value, manifest, command, schedule_changed=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not job.exists()


@pytest.mark.parametrize("changed", ["old96", "production", "scope", "no_context",
    "resume", "pooling", "empty_cohort", "relative_path", "missing_go"])
def test_registration_scope_and_go_required_before_any_dispatch(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    if changed == "old96":
        value["schedule"] = dict(cases=8, methods=4, repetitions=3, total_runs=96)
    elif changed == "production":
        value["expected_production_source_sha256"] = "different"
    elif changed == "scope":
        value["study_semantics"] = "resume_or_pool"
    elif changed == "no_context":
        value["protected_prior_contexts"] = []
    elif changed == "resume":
        value["protected_prior_contexts"][0]["no_resume"] = False
    elif changed == "pooling":
        value["protected_prior_contexts"][0]["no_pooling"] = False
    elif changed == "empty_cohort":
        value["cohort_id"] = ""
    elif changed == "relative_path":
        value["driver"] = "bundle/dummy_driver.py"
    else:
        Path(value["root_go"]).unlink()
    manifest.write_text(json.dumps(value))
    command[-1] = digest(manifest)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not job.exists()


@pytest.mark.parametrize("changed", ["total_runs", "production_source_sha256", "methods", "threads"])
def test_rehashed_protocol_cannot_change_fixed_scope(tmp_path, changed):
    value, manifest, job, command = fixture(tmp_path)
    path = Path(value["protocol"])
    metadata = json.loads(path.read_text())
    metadata[changed] = (96 if changed == "total_runs" else {"OMP_NUM_THREADS":"2"}
        if changed == "threads" else [] if changed == "methods" else "different")
    path.write_text(json.dumps(metadata))
    rebind(value, manifest, command, protocol_changed=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0 and not job.exists()


@pytest.mark.parametrize("injected", ["child_started_write", "wait"])
def test_supervisor_post_spawn_failures_reap_actual_dummy(tmp_path, injected):
    _value, manifest, job, _command = fixture(tmp_path, seconds=10)
    job.mkdir()
    (job / "registration.json").write_bytes(manifest.read_bytes())
    wrapper = tmp_path / "supervisor_fault_wrapper.py"
    wrapper.write_text("import importlib.util,sys\n"
        "from pathlib import Path\n"
        f"s=importlib.util.spec_from_file_location('private_supervisor_fault',{str(LAUNCHER)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        f"injected={injected!r}\n"
        "if injected=='child_started_write':\n"
        " original=m.record\n"
        " def faulty_record(path,value):\n"
        "  if path.name=='child_started.json':raise OSError('manufactured child-start persistence failure')\n"
        "  return original(path,value)\n"
        " m.record=faulty_record\n"
        "else:\n"
        " original=m.subprocess.Popen\n"
        " class WaitFault:\n"
        "  def __init__(self,child):self.child=child;self.pid=child.pid;self.first=True\n"
        "  def poll(self):return self.child.poll()\n"
        "  def wait(self,*args,**kwargs):\n"
        "   if self.first:self.first=False;raise OSError('manufactured wait failure')\n"
        "   return self.child.wait(*args,**kwargs)\n"
        " def faulty_popen(*args,**kwargs):return WaitFault(original(*args,**kwargs))\n"
        " m.subprocess.Popen=faulty_popen\n"
        f"sys.exit(m.supervise(Path({str(job)!r}),{digest(manifest)!r},{digest(LAUNCHER)!r}))\n")
    result = subprocess.run([sys.executable,"-B","-I",str(wrapper)],capture_output=True,
        text=True,timeout=5,check=False)
    assert result.returncode == 2
    failure = wait_json(job / "supervisor_failure.json")
    assert failure["cleanup"] == "known_child_reaped"
    assert failure["actual_child_returncode"] == -signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.kill(failure["child_PID"],0)
    assert not (job / "child_exit.json").exists()


def test_immutable_records_never_replace_original(tmp_path):
    spec = importlib.util.spec_from_file_location("prior48_record_control", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "event.json"
    module.record(path,{"event":"first"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.record(path,{"event":"second"})
    assert path.read_bytes() == original
