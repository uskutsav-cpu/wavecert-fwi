#!/usr/bin/env python3
"""Versioned prior48 detached stdlib supervisor. Default CLI launches nothing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PRODUCTION_SHA256 = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
METHODS = {"normalized_exact_data_only": 0.0, "normalized_exact_fractional_update_prior": 100.0}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    """Immutable, flushed event. Existing records are never replaced."""
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def registration(path):
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 2 or not isinstance(value.get("cohort_id"), str)
            or not value["cohort_id"]):
        raise ValueError("explicit registered cohort required")
    if (value.get("schedule") != dict(cases=8, methods=2, repetitions=3, total_runs=48)
            or any(type(n) is not int for n in value["schedule"].values())):
        raise ValueError("only the full registered48 cohort is supported")
    if (value.get("expected_production_source_sha256") != PRODUCTION_SHA256
            or value.get("expected_production_file_count") != 161
            or value.get("study_semantics") != "new_48_slot_development_no_resume_no_pooling"):
        raise ValueError("explicit production9f and fresh development scope required")
    for key in ("python", "driver", "protocol", "schedule_file", "root_go", "repo", "output"):
        p = Path(value[key])
        if not p.is_absolute() or p != p.resolve():
            raise ValueError("canonical absolute path required: " + key)
    for name in ("driver", "protocol", "schedule_file", "root_go"):
        if sha(Path(value[name])) != value["expected_" + name + "_sha256"]:
            raise ValueError("registered hash mismatch: " + name)
    if not Path(value["python"]).is_file() or not Path(value["repo"]).is_dir():
        raise ValueError("registered interpreter/repository unavailable")
    contexts = value.get("protected_prior_contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("explicit protected prior-development contexts required")
    seen = set()
    for context in contexts:
        for key in ("output", "receipt"):
            previous = Path(context[key])
            if not previous.is_absolute() or previous != previous.resolve():
                raise ValueError("canonical protected context path required")
        previous = Path(context["output"])
        if (previous in seen or not previous.is_dir()
                or context.get("no_resume") is not True or context.get("no_pooling") is not True
                or not isinstance(context.get("role"), str) or not context["role"]
                or sha(Path(context["receipt"])) != context["expected_receipt_sha256"]):
            raise ValueError("protected context identity/semantics mismatch")
        seen.add(previous)
    output = Path(value["output"])
    if output.exists() or not output.parent.is_dir():
        raise ValueError("fresh producer output required; no resume")
    protected = [Path(value["repo"]), *(Path(c["output"]) for c in contexts)]
    protected += [Path(value[k]).parent for k in ("driver", "protocol", "schedule_file")]
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("output must be new and outside repository/protected inputs/context")
    protocol = json.loads(Path(value["protocol"]).read_text())
    if (protocol.get("total_runs") != 48 or len(protocol.get("identities", [])) != 8
            or protocol.get("production_source_sha256") != PRODUCTION_SHA256
            or protocol.get("production_file_count") != 161
            or protocol.get("root_go_authorization") != "GO_CLASSICAL_PRIOR48_DEVELOPMENT"
            or protocol.get("methods") != [{"name": k, "lambda": v} for k, v in METHODS.items()]
            or Path(value["schedule_file"]) != Path(value["protocol"]).with_name("FROZEN48_SCHEDULE.json")
            or protocol.get("schedule_sha256") != value["expected_schedule_file_sha256"]):
        raise ValueError("registration must bind actual prior48 protocol/schedule/production")
    rows = json.loads(Path(value["schedule_file"]).read_text())
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("all48 registered schedule rows required")
    groups = {}
    for index, row in enumerate(rows):
        if (any(type(row.get(k)) is not int for k in ("run_index", "case_index", "repetition", "order_position"))
                or row["run_index"] != index or row["case_index"] not in range(8)
                or row["repetition"] not in range(3) or row["order_position"] not in (0, 1)
                or row.get("method") not in METHODS
                or isinstance(row.get("weight"), bool) or row.get("weight") != METHODS[row["method"]]
                or row.get("identity") != protocol["identities"][row["case_index"]]
                or row.get("status") != "unattempted" or row.get("final_study_eligible") is not False
                or row.get("primary_quality") is not None):
            raise ValueError("registered schedule row differs from untouched prior48 scope")
        group = groups.setdefault((row["case_index"], row["repetition"]), [])
        group.append((row["method"], row["order_position"]))
    if (len(groups) != 24 or any(len(group) != 2 or {m for m, _ in group} != set(METHODS)
            or {position for _, position in group} != {0, 1} for group in groups.values())):
        raise ValueError("complete8cases x2arms x3repetitions required")
    go = json.loads(Path(value["root_go"]).read_text())
    required = {"authorization": "GO_CLASSICAL_PRIOR48_DEVELOPMENT",
        "driver_sha256": value["expected_driver_sha256"],
        "protocol_sha256": value["expected_protocol_sha256"],
        "schedule_sha256": value["expected_schedule_file_sha256"],
        "production_source_sha256": PRODUCTION_SHA256}
    if (any(go.get(k) != expected for k, expected in required.items())
            or go.get("independent_source_review_clear") is not True):
        raise ValueError("exact separate prior48 root source/protocol/schedule GO required")
    if (not isinstance(value["threads"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value["threads"].items())
            or value["threads"] != protocol["threads"]):
        raise ValueError("thread settings must match frozen protocol")
    return value


def child_command(value):
    return [value["python"], "-B", "-I", "-u", value["driver"], "--execute-study",
        "--protocol", value["protocol"], "--repo", value["repo"], "--output", value["output"],
        "--root-go", value["root_go"],
        "--expected-driver-sha256", value["expected_driver_sha256"],
        "--expected-protocol-sha256", value["expected_protocol_sha256"]]


def dispatch(manifest_path, job, expected_manifest, expected_launcher):
    if sha(Path(__file__)) != expected_launcher or sha(manifest_path) != expected_manifest:
        raise ValueError("explicit launcher/registration hash mismatch")
    value = registration(manifest_path)
    immutable_roots = [Path(value[k]) for k in ("repo", "output")]
    immutable_roots += [Path(c["output"]) for c in value["protected_prior_contexts"]]
    immutable_roots += [Path(value[k]).parent for k in ("driver", "protocol", "schedule_file")]
    if (job.exists() or not job.parent.is_dir()
            or any(job.is_relative_to(root) for root in immutable_roots)):
        raise ValueError("fresh separate supervision directory required")
    job.mkdir()
    worker = None
    try:
        record(job / "launch_request.json", dict(created_utc=utc(), dispatch_PID=os.getpid(),
            dispatch_PPID=os.getppid(), launcher_sha256=expected_launcher,
            registration_sha256=expected_manifest, registration=value,
            command=child_command(value), no_project_work_in_dispatch=True))
        # Exact bytes copied, not a normalized JSON surrogate for the frozen file.
        with (job / "registration.json").open("xb") as stream:
            stream.write(manifest_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        with (job / "supervisor.log").open("xb") as log:
            worker = subprocess.Popen([value["python"], "-B", "-I", "-u", str(Path(__file__).resolve()),
                "--supervise", str(job), "--expected-manifest-sha256", expected_manifest,
                "--expected-launcher-sha256", expected_launcher],
                cwd=str(job.parent), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True)
        result = dict(dispatched_utc=utc(), supervisor_PID=worker.pid,
            dispatch_only=True, actual_driver_exit_code=None,
            detached_session=True, stdout_pipe_to_toolhost=False,
            limitation="External SIGKILL/process-tree or machine termination can still prevent final receipts")
        record(job / "dispatch_receipt.json", result)
        return result
    except BaseException as exc:
        cleanup = dict(supervisor_PID=None if worker is None else worker.pid,
            supervisor_returncode=None, child_PID=None, actual_child_returncode=None,
            in_flight_child_unknown=worker is not None)
        if worker is not None:
            try:
                if worker.poll() is None:
                    try:
                        os.kill(worker.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    cleanup["supervisor_returncode"] = worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    cleanup["supervisor_returncode"] = worker.wait()
                if (job / "child_started.json").is_file():
                    cleanup["child_PID"] = json.loads((job / "child_started.json").read_text())["PID"]
                if (job / "child_exit.json").is_file():
                    cleanup["actual_child_returncode"] = json.loads((job / "child_exit.json").read_text())["actual_returncode"]
                    cleanup["in_flight_child_unknown"] = False
                elif not (job / "child_spawn_requested.json").exists():
                    cleanup["in_flight_child_unknown"] = False
            except BaseException as cleanup_exc:
                cleanup["error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        record(job / "dispatch_failure.json", dict(utc=utc(), error=f"{type(exc).__name__}: {exc}",
            project_completion_not_inferred=True, cleanup=cleanup))
        raise


def supervise(job, expected_manifest, expected_launcher):
    child = None
    observed_signals = []
    child_returncode = None

    def on_signal(signum, _frame):
        observed_signals.append(signum)
        delivery = "no_child_started"
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
                delivery = "forwarded_to_child_process_group"
            except ProcessLookupError:
                delivery = "child_group_already_absent"
        record(job / f"signal_{len(observed_signals):03d}.json",
            dict(utc=utc(), signal=signum, delivery=delivery, child_PID=None if child is None else child.pid))

    try:
        if sha(Path(__file__)) != expected_launcher or sha(job / "registration.json") != expected_manifest:
            raise ValueError("supervisor source/registration changed")
        value = registration(job / "registration.json")
        record(job / "supervisor_started.json", dict(utc=utc(), PID=os.getpid(), PPID=os.getppid(),
            session_ID=os.getsid(0), process_group_ID=os.getpgrp(),
            python=sys.version, launcher_sha256=expected_launcher, registration_sha256=expected_manifest))
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, on_signal)
        env = dict(os.environ)
        env.update(value["threads"])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if observed_signals:
            raise InterruptedError("supervisor stop requested before producer start")
        command = child_command(value)
        before = dict(utc=utc(), perf_counter=time.perf_counter(), command=command,
            driver_sha256=sha(Path(value["driver"])), protocol_sha256=sha(Path(value["protocol"])),
            root_go_sha256=sha(Path(value["root_go"])), schedule_sha256=sha(Path(value["schedule_file"])))
        record(job / "child_spawn_requested.json", before)
        with (job / "producer.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=value["repo"], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True)
            record(job / "child_started.json", dict(utc=utc(), PID=child.pid,
                expected_session_and_process_group_ID=child.pid, supervisor_PID=os.getpid(),
                spawn_requested_utc=before["utc"], command=command))
            if observed_signals:
                # Covers a stop request arriving during Popen before assignment.
                try:
                    os.killpg(child.pid, observed_signals[-1])
                except ProcessLookupError:
                    pass
            # No timeout, retries or restarts. This wait observes the actual child.
            child_returncode = child.wait()
        complete = Path(value["output"]) / "complete_receipt.json"
        record(job / "child_exit.json", dict(observed_utc=utc(), child_PID=child.pid,
            actual_returncode=child_returncode,
            signal_if_negative=-child_returncode if child_returncode < 0 else None,
            complete_receipt_exists=complete.is_file(),
            complete_receipt_sha256=sha(complete) if complete.is_file() else None,
            process_exit_is_not_independent_study_integrity_validation=True))
        record(job / "supervisor_exit.json", dict(utc=utc(), status="child_exit_observed",
            supervisor_PID=os.getpid(), child_PID=child.pid, actual_child_returncode=child_returncode,
            observed_signals=observed_signals))
        return 0
    except BaseException as exc:
        # If a post-spawn persistence error occurs, stop/reap our known child
        # rather than leaving an untracked study running. Never invent its code.
        cleanup = None
        if child is not None and child_returncode is None:
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    child_returncode = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child_returncode = child.wait()
                cleanup = "known_child_reaped"
            except BaseException as cleanup_exc:
                cleanup = f"unknown: {type(cleanup_exc).__name__}: {cleanup_exc}"
        record(job / "supervisor_failure.json", dict(utc=utc(), error=f"{type(exc).__name__}: {exc}",
            supervisor_PID=os.getpid(), child_PID=None if child is None else child.pid,
            actual_child_returncode=child_returncode, cleanup=cleanup,
            absent_exit_receipt_means_unknown=True))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job-directory", type=Path)
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-launcher-sha256")
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise.resolve(), args.expected_manifest_sha256, args.expected_launcher_sha256)
    if not args.launch:
        print(json.dumps(dict(status="metadata_only_no_launch", detached_supervisor=True,
            fixed_full_cohort=48, default_project_work=False, separate_root_go_required=True)))
        return 0
    if any(getattr(args, k) is None for k in ("manifest", "job_directory", "expected_manifest_sha256", "expected_launcher_sha256")):
        parser.error("explicit registration/job directory and both hashes required")
    result = dispatch(args.manifest.resolve(), args.job_directory.resolve(),
        args.expected_manifest_sha256, args.expected_launcher_sha256)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
