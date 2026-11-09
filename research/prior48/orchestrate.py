"""Held two-phase GitHub orchestration; default launches nothing.

Control mode runs only four fresh manufactured-control subprocesses. Study mode
requires a reviewed registration-only child of a successful control commit.
No outcomes are interpreted, and this runner has no repository-write operation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY = "uskutsav-cpu/wavecert-fwi"
BRANCH = "research-prior48-remote-01"
BASELINE = "1983664d5e033dab1b131b80d9aaf2358fc6c5de"
PRODUCTION = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
REGISTRATION = "research/prior48/RUN_REGISTRATION_01.json"
WORKFLOW = ".github/workflows/ci.yml"
WORKFLOW_NAME = "Prior48 remote development"
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
CONTROL_GROUPS = (
    ("prior", "controls/prior/test_fractional_velocity_prior.py", 39),
    ("adapter", "controls/adapter/test_composed_exact_adapter.py", 21),
    ("driver", "driver/test_driver_manufactured.py", 30),
    ("launcher", "launcher/test_prior48_launcher.py", 50),
)
STUDY_CAP = 24 * 1024**2
TOTAL_ARTIFACT_CAP = 32 * 1024**2
TOKEN_NAMES = ("GITHUB_TOKEN", "GH_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN")


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(value))


def git(path, *arguments):
    return subprocess.run(["git", "-C", str(path), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def source_contract(root):
    paths = [root / "pyproject.toml", *root.glob("requirements*.txt"), *(root / "src").rglob("*.py"),
        *(p for p in (root / "configs").rglob("*") if p.is_file()),
        *(p for p in (root / "scripts").rglob("*") if p.suffix in {".py", ".sh"}),
        *(root / ".github/workflows").glob("*.yml")]
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in sorted(set(paths)) if p.is_file()}
    digest = hashlib.sha256()
    for name, value in manifest.items():
        digest.update(name.encode() + b"\0" + value.encode() + b"\0")
    if digest.hexdigest() != PRODUCTION or len(manifest) != 161:
        raise ValueError("baseline production9f/161 source contract differs")
    return {"sha256": digest.hexdigest(), "files": manifest}


def verify_bundle(experiment, registration):
    path = experiment / "research/prior48/BUNDLE_MANIFEST.json"
    if sha(path) != registration["bundle_manifest_sha256"]:
        raise ValueError("registered bundle manifest changed")
    manifest = json.loads(path.read_bytes())
    if manifest["baseline_commit"] != BASELINE or manifest["production_source_sha256"] != PRODUCTION:
        raise ValueError("bundle baseline differs")
    for relative, expected in manifest["files"].items():
        source = experiment / relative
        if (source.is_symlink() or not source.resolve().is_relative_to(experiment)
                or sha(source) != expected):
            raise ValueError("bundle source/hash differs: " + relative)
    if REGISTRATION in manifest["files"] or "research/prior48/BUNDLE_MANIFEST.json" in manifest["files"]:
        raise ValueError("registration/manifest must not hash themselves")
    return manifest


def validate_admission(registration, state, manifest, previous=None, control_run=None):
    """Pure fail-closed gate; it creates no process, model or network request."""
    if (state["repository"] != REPOSITORY or state["ref"] != "refs/heads/" + BRANCH
            or state["event"] != "push" or state["attempt"] != 1 or not state["public"]
            or state["forced"] or state["deleted"] or not re.fullmatch("[0-9a-f]{40}", state["sha"])):
        raise ValueError("only first-attempt standard public dedicated-branch push is admitted")
    required = {"schema_version": 1, "baseline_commit": BASELINE,
        "production_source_sha256": PRODUCTION, "production_file_count": 161,
        "expected_control_counts": {name: count for name, _file, count in CONTROL_GROUPS},
        "python_version": "3.11.18", "numpy_version": "2.4.6", "scipy_version": "1.17.1"}
    if any(registration.get(k) != v for k, v in required.items()):
        raise ValueError("registration fixed source/runtime/control scope differs")
    authorized = registration.get("execution_authorized")
    if authorized is False:
        if (registration.get("mode") != "controls_only" or registration.get("root_go") is not None
                or registration.get("root_go_sha256") is not None
                or registration.get("source_bundle_commit") not in (None, state["sha"])
                or registration.get("prior_successful_control") is not None):
            raise ValueError("control-only registration must contain no study GO")
        return "controls_only"
    if authorized is not True or registration.get("mode") != "study":
        raise ValueError("explicit Boolean root activation required")
    bundle_commit = registration.get("source_bundle_commit")
    if (not isinstance(bundle_commit, str) or not re.fullmatch("[0-9a-f]{40}", bundle_commit)
            or state["parents"] != [bundle_commit] or state["changed_paths"] != [REGISTRATION]):
        raise ValueError("activation must be registration-only direct child of the source/control commit")
    prior = registration.get("prior_successful_control")
    if (not isinstance(prior, dict) or prior.get("commit") != bundle_commit
            or type(prior.get("run_id")) is not int or prior["run_id"] <= 0
            or prior.get("run_attempt") != 1):
        raise ValueError("exact prior first-attempt successful control run required")
    if (previous is None or previous.get("execution_authorized") is not False
            or previous.get("mode") != "controls_only"
            or previous.get("bundle_manifest_sha256") != registration["bundle_manifest_sha256"]):
        raise ValueError("source commit registration was not the same control-only bundle")
    if control_run is None:
        return "needs_control_run_verification"
    if (control_run.get("id") != prior["run_id"] or control_run.get("run_attempt") != 1
            or control_run.get("status") != "completed" or control_run.get("conclusion") != "success"
            or control_run.get("head_sha") != bundle_commit or control_run.get("head_branch") != BRANCH
            or control_run.get("event") != "push" or control_run.get("name") != WORKFLOW_NAME
            or control_run.get("path", "").split("@", 1)[0] != WORKFLOW
            or control_run.get("repository", {}).get("full_name") != REPOSITORY
            or control_run.get("head_repository", {}).get("full_name") != REPOSITORY):
        raise ValueError("GitHub did not confirm the registered control-only workflow success")
    go = registration.get("root_go")
    runtime = manifest["runtime"]
    required_go = {"authorization": "GO_CLASSICAL_PRIOR48_DEVELOPMENT",
        "driver_sha256": runtime["driver_sha256"], "protocol_sha256": runtime["protocol_sha256"],
        "schedule_sha256": runtime["schedule_sha256"], "production_source_sha256": PRODUCTION}
    if (not isinstance(go, dict) or any(go.get(k) != v for k, v in required_go.items())
            or go.get("independent_source_review_clear") is not True
            or hashlib.sha256(encoded(go)).hexdigest() != registration.get("root_go_sha256")):
        raise ValueError("exact separately registered source/protocol/schedule root GO required")
    return "study"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def control_run_metadata(run_id):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("read-only Actions token unavailable for prior-run verification")
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{run_id}",
        headers={"Accept": "application/vnd.github+json", "Authorization": "Bearer " + token,
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
        if response.status != 200 or response.geturl() != request.full_url:
            raise ValueError("unexpected prior-run API endpoint/status")
        raw = response.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("prior-run metadata exceeds small response cap")
    return json.loads(raw)


def controlled_environment():
    result = dict(os.environ)
    for name in TOKEN_NAMES:
        result.pop(name, None)
    result.update(THREADS)
    result.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                  PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", MPLBACKEND="Agg")
    return result


def resources():
    memory = {}
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable"):
                memory[name] = int(value.split()[0]) * 1024
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory.get("MemTotal"), "available_memory_bytes": memory.get("MemAvailable"),
        "free_disk_bytes": shutil.disk_usage(Path.cwd()).free, "load_average": list(os.getloadavg())}


def verify_xml(path, expected):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {k: sum(int(s.attrib.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    cases = [case for suite in suites for case in suite.findall("testcase")]
    identities = [(c.attrib.get("classname"), c.attrib.get("name")) for c in cases]
    if counts != {"tests": expected, "failures": 0, "errors": 0, "skipped": 0} or len(cases) != expected or len(set(identities)) != expected:
        raise ValueError("actual JUnit counts or unique collected controls differ")
    return counts


def run_controls(bundle, live):
    rows = []
    controls = live / "controls"
    controls.mkdir()
    (live / "control_tmp").mkdir()
    for name, relative, expected in CONTROL_GROUPS:
        log, xml = controls / f"{name}.log", controls / f"{name}.xml"
        command = [sys.executable, "-B", "-I", "-m", "pytest", "-q", "-W", "error", "-p", "no:cacheprovider",
            "--basetemp", str(live / "control_tmp" / name), "--junitxml", str(xml), str(bundle / relative)]
        before = time.perf_counter()
        print(json.dumps({"phase": "manufactured_controls", "group": name, "expected_tests": expected}), flush=True)
        with log.open("xb") as stream:
            result = subprocess.run(command, cwd=bundle / Path(relative).parent,
                env=controlled_environment(), stdout=stream, stderr=subprocess.STDOUT, check=False)
        row = {"group": name, "actual_exit_code": result.returncode, "seconds": time.perf_counter()-before,
               "command": command, "log_sha256": sha(log), "project_arrays_or_PDE": False}
        record(controls / f"{name}_execution.json", row)
        if result.returncode != 0:
            raise RuntimeError("manufactured control subprocess failed: " + name)
        row["JUnit"] = verify_xml(xml, expected)
        row["JUnit_sha256"] = sha(xml)
        rows.append(row)
    record(live / "metadata/control_summary.json", {"status": "FOUR_FRESH_CONTROL_SUBPROCESSES_PASS",
        "total_tests": sum(row["JUnit"]["tests"] for row in rows), "groups": rows,
        "project_arrays_decoded": 0, "project_PDE_calls": 0})


def study_registration(bundle, baseline, live, registration, manifest):
    inputs = live / "runtime_inputs"
    inputs.mkdir()
    go = inputs / "root_go.json"
    with go.open("xb") as stream:
        stream.write(encoded(registration["root_go"]))
    if sha(go) != registration["root_go_sha256"]:
        raise ValueError("materialized root GO bytes differ")
    contexts = []
    for cohort, name in (("cohort_01", "cohort_01_external_observation.json"),
                          ("cohort_02", "cohort_02_terminal_observation.json")):
        directory = baseline / "results/frequency_continuation_development"
        receipt = directory / name
        contexts.append({"output": str((directory / cohort).resolve()), "receipt": str(receipt.resolve()),
            "expected_receipt_sha256": sha(receipt), "no_resume": True, "no_pooling": True,
            "role": "Protected prior inspected development context only; never resumed or pooled"})
    runtime = manifest["runtime"]
    protocol = bundle / "driver/protocol.json"
    p = json.loads(protocol.read_bytes())
    value = {"schema_version": 2, "cohort_id": "prior48_remote_01",
        "schedule": {"cases": 8, "methods": 2, "repetitions": 3, "total_runs": 48},
        "python": str(Path(sys.executable).resolve()), "driver": str((bundle / "driver/run_prior48.py").resolve()),
        "protocol": str(protocol.resolve()), "schedule_file": str((bundle / "driver/FROZEN48_SCHEDULE.json").resolve()),
        "root_go": str(go.resolve()), "repo": str(baseline), "output": str((live / "study").resolve()),
        "expected_driver_sha256": runtime["driver_sha256"], "expected_protocol_sha256": runtime["protocol_sha256"],
        "expected_schedule_file_sha256": runtime["schedule_sha256"], "expected_root_go_sha256": sha(go),
        "expected_production_source_sha256": PRODUCTION, "expected_production_file_count": 161,
        "study_semantics": "new_48_slot_development_no_resume_no_pooling",
        "protected_prior_contexts": contexts, "threads": p["threads"]}
    if p["threads"] != THREADS or p["disk"]["maximum_output_bytes"] != STUDY_CAP:
        raise ValueError("frozen study thread/output contract differs")
    path = inputs / "launch_registration.json"
    record(path, value)
    return path


def recover_supervisor_pid(job, returned_dispatch, assigned_pid):
    """Recover only an explicitly recorded owned handle, never guess a PID."""
    if type(assigned_pid) is int and assigned_pid > 1:
        return assigned_pid, "assigned_supervisor_PID"
    if isinstance(returned_dispatch, dict):
        pid = returned_dispatch.get("supervisor_PID")
        if type(pid) is int and pid > 1:
            return pid, "returned_dispatch"
    path = job / "dispatch_receipt.json"
    if path.is_file():
        value = json.loads(path.read_bytes())
        pid = value.get("supervisor_PID")
        if (type(pid) is int and pid > 1 and value.get("dispatch_only") is True
                and value.get("detached_session") is True):
            return pid, "immutable_dispatch_receipt"
    return None, "no_verified_handle"


def run_study(bundle, baseline, live, registration, manifest):
    path = study_registration(bundle, baseline, live, registration, manifest)
    launcher = bundle / "launcher/launch_prior48.py"
    spec = importlib.util.spec_from_file_location("_exact_prior48_supervisor", launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha(launcher) != manifest["runtime"]["launcher_sha256"]:
        raise ValueError("frozen tested supervisor changed")
    stopped, supervisor_pid, phase = [], None, "dispatch"

    def on_signal(signum, _frame):
        stopped.append(signum)
        if phase == "dispatch":
            raise KeyboardInterrupt("runner stop during supervised dispatch")
        if supervisor_pid is not None:
            try:
                os.kill(supervisor_pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, on_signal)
    # Token needed only by the already completed read-only API gate. Do not pass
    # repository/runtime tokens into the frozen supervisor or numerical child.
    for name in TOKEN_NAMES:
        os.environ.pop(name, None)
    job = live / "supervision"
    dispatch = None
    try:
        dispatch = module.dispatch(path, job, sha(path), sha(launcher))
        supervisor_pid, phase = dispatch["supervisor_PID"], "monitor"
        record(live / "metadata/dispatch_observation.json", dispatch)
        cursor = 0
        while True:
            producer = job / "producer.log"
            if producer.is_file():
                with producer.open("rb") as stream:
                    stream.seek(cursor)
                    block = stream.read(65536)
                    cursor += len(block)
                if block:
                    sys.stdout.buffer.write(block)
                    sys.stdout.buffer.flush()
            if (job / "supervisor_failure.json").is_file():
                raise RuntimeError("frozen supervisor reported failure; preserved actual prefix")
            if (job / "child_exit.json").is_file() and (job / "supervisor_exit.json").is_file():
                child = json.loads((job / "child_exit.json").read_bytes())
                record(live / "metadata/actual_child_exit_observation.json", {"supervisor_PID": supervisor_pid,
                    "actual_child_exit": child, "observed_stop_signals": stopped,
                    "no_scientific_outcomes_interpreted": True, "study_integrity_validated": False})
                if child["actual_returncode"] != 0 or stopped:
                    raise RuntimeError("actual numerical child nonzero/stopped; no retry")
                return
            try:
                os.kill(supervisor_pid, 0)
            except ProcessLookupError:
                raise RuntimeError("supervisor handle missing without terminal receipts; no restart") from None
            time.sleep(.25)
    except BaseException:
        phase = "cleanup"
        supervisor_pid, recovery_source = recover_supervisor_pid(job, dispatch, supervisor_pid)
        # Coordinate error-stop through the same tested supervisor. A missing
        # terminal receipt remains unknown, never an invented child exit.
        sent = False
        if supervisor_pid is not None and not (job / "supervisor_exit.json").is_file():
            try:
                os.kill(supervisor_pid, signal.SIGTERM)
                sent = True
            except ProcessLookupError:
                pass
            until = time.monotonic() + 10
            while time.monotonic() < until:
                if (job / "supervisor_exit.json").is_file() or (job / "supervisor_failure.json").is_file():
                    break
                time.sleep(.25)
        record(live / "metadata/error_stop_observation.json", {"supervisor_PID":supervisor_pid,
            "supervisor_PID_recovery_source":recovery_source,
            "SIGTERM_sent_to_tested_supervisor":sent,
            "child_exit_receipt_sha256":sha(job/"child_exit.json") if (job/"child_exit.json").is_file() else None,
            "child_exit_unknown":not (job/"child_exit.json").is_file(), "no_restart":True})
        raise


def copy_snapshot(source, target, size):
    """Copy at most the admitted snapshot size; retain a short prefix on error."""
    with source.open("rb") as src, target.open("xb") as dst:
        remaining = size
        while remaining:
            block = src.read(min(65536, remaining))
            if not block:
                break
            dst.write(block)
            remaining -= len(block)


def collect_upload(live, upload, mode):
    """Copy bounded evidence; control temporary arrays and checkouts stay out."""
    upload.mkdir(parents=True, exist_ok=True)
    roots = [live / "metadata", live / "controls"]
    if mode == "study":
        roots += [live / "runtime_inputs", live / "supervision", live / "study"]
    # The workflow may already have bootstrap/install/console logs here even if
    # setup failed. Account for those small files rather than overwriting them.
    retained = [{"path": str(p.relative_to(upload)), "bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(upload.rglob("*")) if p.is_file()]
    used, study_used, omitted = sum(x["bytes"] for x in retained), 0, []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(live):
                omitted.append({"path": str(path.relative_to(live)), "reason": "unsafe source path"})
                continue
            size = path.stat().st_size
            item = {"path": str(path.relative_to(live)), "bytes": size, "sha256": sha(path)}
            study = path.is_relative_to(live / "study")
            if used + size > TOTAL_ARTIFACT_CAP-1024**2 or study and study_used+size > STUDY_CAP:
                omitted.append({**item, "reason": "declared evidence byte budget exceeded"})
                continue
            target = upload / path.relative_to(live)
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_snapshot(path, target, size)
            actual_size, actual_sha = target.stat().st_size, sha(target)
            stable = (actual_size == size and actual_sha == item["sha256"]
                and path.stat().st_size == size and sha(path) == item["sha256"])
            used += actual_size
            study_used += actual_size if study else 0
            retained.append({"path":item["path"],"bytes":actual_size,"sha256":actual_sha,
                "source_snapshot_bytes":size,"source_snapshot_sha256":item["sha256"],
                "copy_and_source_stable":stable})
            if not stable:
                omitted.append({**item,"reason":"source mutated or bounded copied bytes differ; retained target hash is authoritative"})
    inventory = {"retained_bytes": used, "study_bytes": study_used,
        "study_cap_bytes": STUDY_CAP, "total_cap_bytes": TOTAL_ARTIFACT_CAP,
        "retained": retained, "omitted": omitted, "complete_evidence_collection": not omitted,
        "control_temp_arrays_or_baseline_data_uploaded": False}
    if len(encoded(inventory)) > 1024**2 or used + len(encoded(inventory)) > TOTAL_ARTIFACT_CAP or study_used > STUDY_CAP:
        inventory["complete_evidence_collection"] = False
        omitted.append({"reason":"final actual retained byte total exceeds registered cap"})
    record(upload / "UPLOAD_INVENTORY.json", inventory)
    if omitted:
        raise ValueError("evidence collection incomplete; omission manifest retained")


def execute(experiment, baseline, artifacts):
    if artifacts.is_relative_to(experiment) or artifacts.is_relative_to(baseline):
        raise ValueError("evidence must remain outside both frozen checkouts")
    live, upload = artifacts / "live", artifacts / "upload"
    live.mkdir(parents=True, exist_ok=False)
    (live / "metadata").mkdir()
    mode, result = "controls_only", 2
    began = time.perf_counter()
    try:
        registration = json.loads((experiment / REGISTRATION).read_bytes())
        bundle = experiment / "research/prior48"
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
        commits = git(experiment, "rev-list", "--parents", "-n", "1", "HEAD").split()
        state = {"repository": os.environ.get("GITHUB_REPOSITORY"), "ref": os.environ.get("GITHUB_REF"),
            "event": os.environ.get("GITHUB_EVENT_NAME"), "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
            "sha": os.environ.get("GITHUB_SHA", ""), "parents": commits[1:], "changed_paths": [],
            "public": event.get("repository", {}).get("private") is False,
            "forced": event.get("forced") is not False, "deleted": event.get("deleted") is not False}
        if os.environ.get("GITHUB_ACTIONS") != "true" or commits[0] != state["sha"] or git(baseline,"rev-parse","HEAD") != BASELINE:
            raise ValueError("actual hosted event/source/baseline checkout differs")
        manifest = verify_bundle(experiment, registration)
        production_before = source_contract(baseline)
        record(live / "metadata/source_before.json", {"production": production_before, "bundle": manifest})
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pytest", "h5py")}
        record(live / "metadata/runtime.json", {"python": platform.python_version(), "executable": sys.executable,
            "packages": versions, "threads": {k: os.environ.get(k) for k in THREADS}, "resources": resources(),
            "workflow_setup_scope": "Checkouts/setup-python/pip precede this process; their costs remain in workflow logs, not free project work."})
        if (platform.python_version() != "3.11.18" or versions["numpy"] != "2.4.6" or versions["scipy"] != "1.17.1"
                or any(os.environ.get(k) != v for k,v in THREADS.items())):
            raise ValueError("registered runtime or controlled thread environment differs")
        previous, api = None, None
        if registration.get("execution_authorized") is True:
            state["changed_paths"] = git(experiment,"diff","--name-only",state["parents"][0],"HEAD").splitlines() if len(state["parents"])==1 else []
            parent = registration.get("source_bundle_commit", "")
            if re.fullmatch("[0-9a-f]{40}", parent):
                previous = json.loads(git(experiment,"show",parent+":"+REGISTRATION))
            validate_admission(registration,state,manifest,previous)
            api = control_run_metadata(registration["prior_successful_control"]["run_id"])
            record(live / "metadata/prior_control_run_api.json", api)
        mode = validate_admission(registration,state,manifest,previous,api)
        record(live / "metadata/admission.json", {"mode":mode,"state":state,"registration":registration,
            "source_bundle_commit":state["sha"] if mode=="controls_only" else registration["source_bundle_commit"],
            "run_id":os.environ.get("GITHUB_RUN_ID"),"registration_sha256":sha(experiment/REGISTRATION)})
        if mode == "controls_only":
            run_controls(bundle,live)
        else:
            run_study(bundle,baseline,live,registration,manifest)
        if source_contract(baseline) != production_before or verify_bundle(experiment,registration) != manifest:
            raise ValueError("frozen source changed during job")
        record(live / "metadata/terminal.json", {"status":"orchestration_completed", "mode":mode,
            "process_elapsed_seconds":time.perf_counter()-began,"resources":resources(),
            "scientific_outcomes_interpreted":False,"source_unchanged":True})
        result = 0
    except BaseException as exc:
        record(live / "metadata/failure.json", {"status":"orchestration_failed_or_interrupted", "mode":mode,
            "error":f"{type(exc).__name__}: {exc}","process_elapsed_seconds":time.perf_counter()-began,
            "no_retry_or_fallback":True,"scientific_success_not_inferred":True})
    finally:
        try:
            collect_upload(live,upload,mode)
        except BaseException as exc:
            result = 2
            print(f"Artifact collection failed: {type(exc).__name__}: {exc}", file=sys.stderr,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    for name in ("experiment","baseline","artifacts"):
        parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    if not args.run:
        print(json.dumps({"status":"HELD_METADATA_ONLY_NO_LAUNCH", "actual_study_authorized":False,
            "required_control_counts":{name:count for name,_file,count in CONTROL_GROUPS}}))
        return 0
    if not all((args.experiment,args.baseline,args.artifacts)):
        parser.error("explicit separate checkout and evidence paths required")
    return execute(args.experiment.resolve(),args.baseline.resolve(),args.artifacts.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
