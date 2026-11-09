"""Held foreground23-slot controller. Default launches no process or numerics."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checkpoint23_controller_common", ROOT/"pilot23_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def utc():
    return datetime.now(timezone.utc).isoformat()


class StopState:
    def __init__(self):
        self.signals = []
        self.child = None

    def handler(self, signum, _frame):
        # Do not raise in a Popen handoff or perform reentrant evidence writes.
        delivery = "no_assigned_child_handle"
        child = self.child
        if child is not None and child.poll() is not None:
            delivery = "assigned_child_already_terminal"
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
                delivery = "forwarded_to_owned_child_group"
            except ProcessLookupError:
                delivery = "owned_group_already_absent"
            except OSError as error:
                delivery = "forward_error:"+str(error)
        self.signals.append({"signal": signum, "utc": utc(), "delivery": delivery,
                             "PID": None if child is None else child.pid})


class SlotError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def tree_bytes(root):
    total = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlink in retained output tree")
            if path.is_file():
                total += path.stat().st_size
    return total


def system_available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])*1024
    raise ValueError("Linux MemAvailable unavailable; no inferred resource fallback")


def resources_before_spawn(output, policy):
    retained = tree_bytes(output)
    free = shutil.disk_usage(output).free
    memory = system_available_memory()
    if (retained > policy["total_output_cap_bytes"]
            or free < policy["total_output_cap_bytes"]-retained+policy["minimum_free_disk_headroom_bytes"]
            or memory < policy["minimum_available_memory_bytes"]):
        raise ValueError("registered disk/memory reservation unavailable before next child")
    return {"retained_output_bytes": retained, "free_disk_bytes": free,
            "available_memory_bytes": memory, "sampling_not_a_hard_allocation_guarantee": True}


def process_group_snapshot(group):
    """Linux selected live-group RSS/CPU sample, not a peak or full process trace."""
    if not Path("/proc").is_dir():
        return {"available": False, "scope": "Linux/proc unavailable"}
    members = []
    page, ticks = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_CLK_TCK")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry/"stat").read_text()
            fields = text[text.rfind(")")+2:].split()
            if int(fields[2]) != group:
                continue
            members.append({"PID": int(entry.name), "process_group_ID": int(fields[2]),
                "session_ID": int(fields[3]), "start_ticks": int(fields[19]),
                "RSS_bytes": int(fields[21])*page,
                "CPU_seconds": (int(fields[11])+int(fields[12]))/ticks})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {"available": True, "members": members,
            "group_RSS_bytes": sum(row["RSS_bytes"] for row in members),
            "live_member_CPU_seconds": sum(row["CPU_seconds"] for row in members),
            "scope": "Sample of current Linux process group; exited members and between-sample peaks are not reconstructed."}


def cpu_children():
    value = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"user_seconds": value.ru_utime, "system_seconds": value.ru_stime,
            "cumulative_peak_RSS_native_units": value.ru_maxrss}


def controller_resources():
    value = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024 if sys.platform.startswith("linux") else 1
    return {"user_CPU_seconds_since_process_start": value.ru_utime,
        "system_CPU_seconds_since_process_start": value.ru_stime,
        "peak_RSS_bytes": int(value.ru_maxrss)*scale,
        "scope": "Controller only, including hash/resource/comparison work so far; separate from reaped native worker families."}


def native_prefix(worker_output, cap=8*1024**2):
    """Preserve known attempt/return metadata without guessing partial kernel work."""
    path = worker_output/"events.jsonl"
    if not path.exists():
        return {"status": "NO_WORKER_NATIVE_EVENT_FILE", "unknown_partial_native_steps": True,
                "unknown_partial_checkpoint_copy_work": True,
                "native_counter_scope": "No event file does not prove that no setup/import work occurred."}
    if path.is_symlink() or path.stat().st_size > cap:
        return {"status": "EVENT_PREFIX_OVER_CAP_OR_SYMLINK", "unknown_partial_native_steps": True,
                "unknown_partial_checkpoint_copy_work": True}
    raw = path.read_bytes()
    known, invalid_tail = [], False
    for line in raw.splitlines():
        try:
            known.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid_tail = True
            break
    counts = {"native_forward_attempted_steps": 0, "native_forward_returned_steps": 0,
        "native_reverse_attempted_steps": 0, "native_reverse_returned_steps": 0,
        "replay_forward_attempted_steps": 0, "replay_forward_returned_steps": 0,
        "checkpoint_copy_attempted_bytes": 0, "checkpoint_copy_returned_bytes": 0}
    attempted, returned, raised = 0, 0, 0
    copy_attempted, copy_returned, copy_raised = 0, 0, 0
    for row in known:
        kind = row.get("kind")
        if kind in ("native_attempt", "native_returned"):
            role = row.get("role")
            suffix = "attempted" if kind == "native_attempt" else "returned"
            if role in ("forward", "reverse"):
                counts[f"native_{role}_{suffix}_steps"] += int(row.get("native_steps", 0))
            if role == "forward" and row.get("phase") == "reverse_with_paid_replay":
                counts[f"replay_forward_{suffix}_steps"] += int(row.get("native_steps", 0))
            attempted += kind == "native_attempt"
            returned += kind == "native_returned"
        elif kind == "native_raised":
            raised += 1
        elif kind in ("checkpoint_copy_attempt", "checkpoint_copy_returned"):
            suffix = "attempted" if kind == "checkpoint_copy_attempt" else "returned"
            counts[f"checkpoint_copy_{suffix}_bytes"] += int(row.get("bytes", 0))
            copy_attempted += kind == "checkpoint_copy_attempt"
            copy_returned += kind == "checkpoint_copy_returned"
        elif kind == "checkpoint_copy_raised":
            copy_raised += 1
    return {"status": "OBSERVED_BOUNDED_WORKER_EVENT_PREFIX", "events_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "parsed_event_records": len(known), "invalid_or_partial_tail": invalid_tail,
        "native_attempted_calls": attempted, "native_returned_calls": returned, "native_raised_calls": raised,
        "unknown_partial_native_steps": invalid_tail or attempted != returned,
        "checkpoint_copy_attempted_calls": copy_attempted,
        "checkpoint_copy_returned_calls": copy_returned, "checkpoint_copy_raised_calls": copy_raised,
        "unknown_partial_checkpoint_copy_work": invalid_tail or copy_attempted != copy_returned or copy_raised > 0,
        "counters": counts, "native_counter_scope": "Observed wrapper metadata only: returned calls give reported completed ranges; unreturned/raised calls have unknown partial executed steps. Setup/import/compilation work is paid external wall/CPU, not inferred as these kernel steps."}


def kill_known_group(child, snapshot):
    """Kill only the still-owned direct child/group; never an arbitrary PID list."""
    if child is None:
        return {"action": "no_child_handle"}
    if child.poll() is None:
        try:
            if os.getpgid(child.pid) != child.pid:
                child.kill()
                return {"action": "SIGKILL_known_direct_child_unexpected_group", "unknown_work": True}
            os.killpg(child.pid, signal.SIGKILL)
            return {"action": "SIGKILL_owned_live_child_group", "group_ID": child.pid}
        except ProcessLookupError:
            return {"action": "owned_group_already_absent"}
    members = snapshot.get("members", []) if snapshot.get("available") else []
    if members and all(row["session_ID"] == child.pid for row in members):
        try:
            os.killpg(child.pid, signal.SIGKILL)
            return {"action": "SIGKILL_surviving_owned_session_group", "group_ID": child.pid,
                    "unexpected_descendant_work": True}
        except ProcessLookupError:
            return {"action": "owned_descendant_group_already_absent", "unexpected_descendant_work": True}
    return {"action": "no_owned_live_group_observed", "unknown_work": bool(members)}


def run_one(command, slot_dir, output, policy, stop, slot_id, worker_output,
            resource_probe=process_group_snapshot, evidence_scope="NATIVE_PILOT_PENDING_VALIDATION"):
    """Actual foreground Popen/poll/wait, separate session, retained failure prefix."""
    entered, started_utc = time.perf_counter(), utc()
    cpu_before = cpu_children()
    child, result_code, wait_observed, reason = None, None, False, None
    started_sha = None
    cleanup = {"action": "not_needed"}
    sample_hash = hashlib.sha256()
    prefix, tail = [], deque(maxlen=policy["resource_tail_samples"])
    sample_count, maximum_RSS = 0, 0
    last = {"available": False, "members": []}
    signal_start = len(stop.signals)
    result = {"schema": "checkpoint_pilot23_child_exit_v1", "slot_id": slot_id,
        "worker_output": str(worker_output), "evidence_scope": evidence_scope,
        "PID": None, "actual_returncode": None, "observed_terminal": False,
        "unknown_work": False, "started_utc": started_utc}
    try:
        C.record(slot_dir/"spawn_requested.json", {"utc": started_utc, "command": command,
            "controller_PID": os.getpid(), "slot_id": slot_id, "no_prior_child_reused": True,
            "entry_perf_counter": entered})
        if stop.signals:
            raise InterruptedError("stop requested before child spawn")
        with (slot_dir/"worker_console.log").open("xb") as console:
            child = subprocess.Popen(command, cwd=slot_dir, env=C.controlled_environment(),
                stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True)
            stop.child = child  # handler never raises during this ownership handoff
            result["PID"] = child.pid
            try:
                observed_group, observed_session = os.getpgid(child.pid), os.getsid(child.pid)
            except ProcessLookupError:
                observed_group = observed_session = None
            C.record(slot_dir/"started.json", {"utc": utc(), "PID": child.pid,
                "controller_PID": os.getpid(), "slot_id": slot_id, "command": command,
                "worker_output": str(worker_output),
                "observed_group_ID": observed_group, "observed_session_ID": observed_session,
                "expected_new_session_and_group_ID": child.pid,
                "unobserved_fast_exit_group_is_not_inferred": observed_group is None})
            started_sha = C.sha(slot_dir/"started.json")
            while child.poll() is None:
                if stop.signals:
                    raise InterruptedError("controller stop requested; later slots remain unattempted")
                last = resource_probe(child.pid)
                if not last.get("available"):
                    raise ValueError("required process-group resource observation unavailable")
                sample = {"elapsed_seconds": time.perf_counter()-entered, **last}
                sample_hash.update(C.encoded(sample))
                sample_count += 1
                if len(prefix) < policy["resource_prefix_samples"]:
                    prefix.append(sample)
                tail.append(sample)
                maximum_RSS = max(maximum_RSS, last["group_RSS_bytes"])
                if (maximum_RSS > policy["sampled_live_group_RSS_cap_bytes"]
                        or tree_bytes(slot_dir) > policy["per_slot_output_cap_bytes"]
                        or tree_bytes(output) > policy["total_output_cap_bytes"]
                        or (slot_dir/"worker_console.log").stat().st_size > policy["per_console_cap_bytes"]):
                    raise ValueError("registered observed output or live group RSS cap exceeded")
                time.sleep(policy["poll_seconds"])
            result_code = child.wait()  # actual cached OS exit if poll reaped it first
            wait_observed = True
        last = resource_probe(child.pid)
        if not last.get("available"):
            result["unknown_work"] = True
            raise RuntimeError("terminal owned-group resource observation unavailable")
        if last.get("available") and last.get("members"):
            cleanup = kill_known_group(child, last)
            result["unknown_work"] = True
            raise RuntimeError("unexpected descendants survived the direct child's actual exit")
        if stop.signals:
            raise InterruptedError("stop observed before slot admission completed")
        if result_code != 0:
            reason = "actual_nonzero_or_signal_exit"
    except BaseException as error:
        reason = type(error).__name__+": "+str(error)
        if child is not None:
            try:
                try:
                    last = resource_probe(child.pid)
                except BaseException as probe_error:
                    last = {"available": False, "members": [], "probe_error": str(probe_error)}
                cleanup = kill_known_group(child, last)
                result_code = child.wait()
                wait_observed = True
                result["unknown_work"] = result["unknown_work"] or cleanup.get("unknown_work", False)
                if last.get("members") and result_code == 0:
                    result["unknown_work"] = True
                try:
                    remaining = resource_probe(child.pid)
                except BaseException:
                    remaining = {"available": False, "members": []}
                if not remaining.get("available") or remaining.get("members"):
                    result["unknown_work"] = True
            except BaseException as cleanup_error:
                cleanup = {"error": type(cleanup_error).__name__+": "+str(cleanup_error),
                           "actual_returncode": result_code}
                result["unknown_work"] = True
    finally:
        stop.child = None
    ended = time.perf_counter()
    result.update(actual_returncode=result_code, observed_terminal=wait_observed,
        external_wall_seconds=ended-entered, terminal_utc=utc(), reason=reason,
        started_sha256=started_sha, cleanup=cleanup, observed_signals=stop.signals[signal_start:])
    try:
        cpu_after = cpu_children()
        result.update(resources={"sample_count": sample_count, "ordered_samples_sha256": sample_hash.hexdigest(),
            "prefix": prefix, "tail": list(tail), "maximum_sampled_live_group_RSS_bytes": maximum_RSS,
            "sampling_scope": "Bounded prefix/tail plus commitment over every sample; not continuous RSS or reconstructed vanished-child CPU.",
            "actual_reaped_process_family_user_CPU_seconds": cpu_after["user_seconds"]-cpu_before["user_seconds"],
            "actual_reaped_process_family_system_CPU_seconds": cpu_after["system_seconds"]-cpu_before["system_seconds"],
            "cumulative_RUSAGE_CHILDREN_peak_native_units_after": cpu_after["cumulative_peak_RSS_native_units"],
            "peak_cumulative_not_subtracted_or_added_to_worker_peak": True},
            retained_slot_bytes=tree_bytes(slot_dir), retained_cohort_bytes=tree_bytes(output))
        result["native_event_prefix"] = native_prefix(worker_output) if child is not None else {
            "status": "NO_CHILD_SPAWNED", "unknown_partial_native_steps": False,
            "unknown_partial_checkpoint_copy_work": False,
            "scope": "No Popen handle was created; this slot launched no worker."}
        result["unknown_partial_native_work"] = result["native_event_prefix"]["unknown_partial_native_steps"]
        result["unknown_partial_checkpoint_copy_work"] = result["native_event_prefix"]["unknown_partial_checkpoint_copy_work"]
        if child is not None and not wait_observed:
            result["unknown_work"] = True
        if (result["retained_slot_bytes"] > policy["per_slot_output_cap_bytes"]
                or result["retained_cohort_bytes"] > policy["total_output_cap_bytes"]
                or (slot_dir/"worker_console.log").exists()
                and (slot_dir/"worker_console.log").stat().st_size > policy["per_console_cap_bytes"]):
            result["reason"] = result["reason"] or "terminal_output_cap_exceeded"
        C.record(slot_dir/"exit.json", result)
    except BaseException as error:
        # Keep minimal observed OS truth even if post-wait metadata cannot be read/written.
        result.update(evidence_assembly_error=type(error).__name__+": "+str(error))
        result.setdefault("unknown_partial_native_work", child is not None)
        result.setdefault("unknown_partial_checkpoint_copy_work", child is not None)
        if child is not None and not wait_observed:
            result["unknown_work"] = True
        try:
            print(json.dumps({"status": "EXIT_EVIDENCE_WRITE_FAILED", "slot_id": slot_id,
                "actual_returncode": result_code, "observed_terminal": wait_observed,
                "unknown_work": result["unknown_work"], "error": str(error)}), file=sys.stderr, flush=True)
        except BaseException:
            pass
        raise SlotError("terminal exit evidence could not be retained", result) from error
    return result


def admit(args):
    if (C.sha(args.registration) != args.registration_sha256
            or C.sha(args.root_go) != args.root_go_sha256
            or C.sha(args.environment) != args.environment_sha256):
        raise ValueError("exact registration/rootGO/environment hashes required")
    registration = C.read_json(args.registration)
    protocol = C.verify_worker()
    plan = C.verify_plan(protocol)
    if (registration.get("schema") != "checkpoint_pilot23_registration_v1"
            or registration.get("execution_authorized") is not True
            or not isinstance(registration.get("cohort_id"), str)
            or not registration["cohort_id"].startswith("checkpoint23_")
            or registration.get("scope") != "constant_medium_checkpoint23_no_project_data"
            or registration.get("slot_count") != 23 or registration.get("resume") is not False
            or registration.get("retry") is not False
            or registration.get("worker_source_files") != C.WORKER_SOURCES
            or registration.get("plan_sha256") != C.sha(ROOT/"FROZEN_PLAN23.json")
            or registration.get("policy_sha256") != C.sha(ROOT/"controller_policy.json")):
        raise ValueError("separate truthful complete23 registration required; no48/96/resume scope")
    manifest = C.verify_bundle(registration)
    go = C.read_json(args.root_go)
    required = {"authorization": "GO_CHECKPOINT_PILOT23_V1", "execution_authorized": True,
        "registration_sha256": args.registration_sha256, "environment_sha256": args.environment_sha256,
        "controller_bundle_manifest_sha256": registration["bundle_manifest_sha256"],
        "plan_sha256": registration["plan_sha256"], "scope": registration["scope"],
        "independent_source_review_clear": True, "external_controller_review_clear": True,
        "native_worker_environment_review_clear": True, "criteria_frozen_before_execution": True}
    if any(go.get(key) != value or type(go.get(key)) is not type(value) for key, value in required.items()):
        raise ValueError("exact separate root controller/source/environment/criteria GO required")
    worker_go = go.get("worker_go", {})
    if (worker_go.get("authorization") != "GO_CONSTANT_MEDIUM_CHECKPOINT_PILOT"
            or worker_go.get("execution_authorized") is not True
            or worker_go.get("independent_source_review_clear") is not True
            or worker_go.get("criteria_frozen_before_execution") is not True
            or worker_go.get("source_files") != C.WORKER_SOURCES):
        raise ValueError("root GO must separately and exactly authorize frozen319 native worker")
    environment = C.read_json(args.environment)
    observed = C.verify_environment(environment, protocol)
    output = C.canonical(args.output)
    prefix = C.canonical(environment["prefix"])
    if (output.exists() or not output.parent.is_dir() or output.is_relative_to(ROOT)
            or output.is_relative_to(prefix)):
        raise ValueError("fresh separate output outside source/runtime required")
    return registration, protocol, plan, manifest, go, environment, observed


def child_command(slot, admission, slot_dir):
    value = [sys.executable, "-B", "-I", "-u", str(ROOT/"worker_entry.py"), "--execute",
        "--environment", str(admission/"environment.json"),
        "--environment-sha256", C.sha(admission/"environment.json"), "--slot-dir", str(slot_dir),
        "--worker-go", str(admission/"worker_go.json"),
        "--worker-go-sha256", C.sha(admission/"worker_go.json"), "--phase", slot["phase"],
        "--slot-id", slot["slot_id"]]
    for field in ("checkpoints", "direction", "epsilon", "sign"):
        if slot[field] is not None:
            value.extend(["--"+field, str(slot[field])])
    return value


def validate_returned(slot, slot_dir, exited, environment, worker_go_sha):
    if (exited["actual_returncode"] != 0 or exited["observed_terminal"] is not True
            or exited["unknown_work"] is not False or exited["reason"] is not None
            or exited.get("unknown_partial_native_work") is not False
            or exited.get("unknown_partial_checkpoint_copy_work") is not False):
        raise ValueError("actual observed normal exit without unknown work required")
    identity = C.read_json(slot_dir/"import_identity.json")
    if (identity.get("status") != "SELECTED_ACTUAL_IMPORT_ORIGINS_MATCH_APPROVED_ENVIRONMENT"
            or identity.get("slot_id") != slot["slot_id"]
            or identity.get("worker_output") != str(slot_dir/"worker")
            or identity.get("PID") != exited["PID"] or identity.get("PPID") != os.getpid()
            or identity.get("process_group_ID") != exited["PID"] or identity.get("session_ID") != exited["PID"]
            or identity.get("worker_go_sha256") != worker_go_sha
            or identity.get("selected_modules") != environment["modules"]):
        raise ValueError("actual child import/PID/session identity differs")
    worker = C.read_json(slot_dir/"worker/result.json")
    if (worker.get("source_files") != C.WORKER_SOURCES or worker.get("root_go_sha256") != worker_go_sha
            or worker.get("status") != "WORKER_RETURNED_NO_SCIENTIFIC_SUCCESS_INFERRED"
            or worker.get("intended_worker_return_code") != 0
            or worker.get("phase") != slot["phase"] or worker.get("checkpoints") != slot["checkpoints"]
            or worker.get("fd_direction") != slot["direction"] or worker.get("fd_epsilon") != slot["epsilon"]
            or worker.get("fd_sign") != slot["sign"]):
        raise ValueError("actual child result is not the exact registered source/slot")
    policy = C.read_json(ROOT/"controller_policy.json")
    if worker["resources"]["worker_peak_RSS_bytes"] > policy["worker_peak_RSS_cap_bytes"]:
        raise ValueError("actual worker-reported Linux peak RSS exceeds fixed review cap")
    return worker


def execute(args):
    entered = time.perf_counter()
    registration, protocol, plan, manifest, go, environment, observed = admit(args)
    policy = C.read_json(ROOT/"controller_policy.json")
    output = C.canonical(args.output)
    output.mkdir()
    admission = output/"admission"
    admission.mkdir()
    for source, name in ((args.registration, "registration.json"), (args.root_go, "root_go.json"),
                         (args.environment, "environment.json")):
        with (admission/name).open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    C.record(admission/"worker_go.json", go["worker_go"])
    worker_go_sha = C.sha(admission/"worker_go.json")
    C.record(admission/"source_environment_observation.json", {"source_bundle": manifest,
        "environment_observation": observed, "controller_PID": os.getpid(),
        "entry_through_admission_seconds": time.perf_counter()-entered,
        "root_go_sha256": args.root_go_sha256, "registration_sha256": args.registration_sha256})
    children = output/"children"
    rows = [{**row, "worker_output": str(children/row["slot_id"]/"worker"),
             "command": child_command(row, admission, children/row["slot_id"])} for row in plan["slots"]]
    C.record(output/"plan.json", {**plan, "slots": rows, "frozen_plan_sha256": C.sha(ROOT/"FROZEN_PLAN23.json"),
        "worker_go_sha256": worker_go_sha, "all23_rows_written_before_any_Popen": True})
    children.mkdir()
    stop, handlers = StopState(), {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, stop.handler)
    completed, failure, unknown = [], None, False
    try:
        for slot in rows:
            if stop.signals:
                raise InterruptedError("stop requested before next slot")
            # Recheck exact immutable source and admission bytes before every Popen.
            C.verify_bundle(registration)
            C.verify_worker()
            if (C.sha(admission/"root_go.json") != args.root_go_sha256
                    or C.sha(admission/"registration.json") != args.registration_sha256
                    or C.sha(admission/"environment.json") != args.environment_sha256
                    or C.sha(admission/"worker_go.json") != worker_go_sha):
                raise ValueError("immutable admission/source changed between slots")
            reservation = resources_before_spawn(output, policy)
            slot_dir = children/slot["slot_id"]
            slot_dir.mkdir()
            C.record(slot_dir/"resource_admission.json", reservation)
            row = {**slot, "state": "attempt_started_terminal_state_unknown", "unknown_work": True}
            completed.append(row)
            try:
                exited = run_one(slot["command"], slot_dir, output, policy, stop,
                    slot["slot_id"], Path(slot["worker_output"]))
            except SlotError as error:
                row.update(error.receipt)
                row["state"] = "failed_exit_evidence"
                unknown = unknown or error.receipt["unknown_work"]
                raise
            state = ("waited_exit0" if exited["actual_returncode"] == 0 else "waited_nonzero") if exited["observed_terminal"] else "no_observed_wait"
            row.update({"state": state,
                "PID": exited["PID"], "actual_returncode": exited["actual_returncode"],
                "external_wall_seconds": exited["external_wall_seconds"], "exit_sha256": C.sha(slot_dir/"exit.json"),
                "started_sha256": exited["started_sha256"], "unknown_work": exited["unknown_work"],
                "unknown_partial_native_work": exited["unknown_partial_native_work"],
                "unknown_partial_checkpoint_copy_work": exited["unknown_partial_checkpoint_copy_work"]})
            unknown = unknown or exited["unknown_work"] or exited["unknown_partial_native_work"] or exited["unknown_partial_checkpoint_copy_work"]
            worker = validate_returned(slot, slot_dir, exited, environment, worker_go_sha)
            row.update(result_sha256=C.sha(slot_dir/"worker/result.json"),
                import_identity_sha256=C.sha(slot_dir/"import_identity.json"), numerical_eligible=False,
                native_worker_selfreported_counters={key: worker[key] for key in
                    ("native_forward", "initial_native_forward", "native_replay_forward_steps", "native_reverse")})
            C.record(slot_dir/"validated.json", row)
    except BaseException as error:
        failure = type(error).__name__+": "+str(error)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    seen = {row["slot_id"] for row in completed}
    unknown_process = any(row.get("unknown_work") is True for row in completed)
    unknown_native = any(row.get("unknown_partial_native_work") is True for row in completed)
    unknown_copy = any(row.get("unknown_partial_checkpoint_copy_work") is True for row in completed)
    unknown = unknown or unknown_process or unknown_native or unknown_copy
    terminal_rows = completed + [{**slot, "state": "unattempted", "actual_returncode": None}
                                 for slot in rows if slot["slot_id"] not in seen]
    all_normal = (failure is None and len(completed) == 23 and not unknown
                  and all(row["state"] == "waited_exit0" and row.get("result_sha256") for row in completed))
    terminal = {"schema": "checkpoint_pilot23_external_v1",
        "status": "ALL_23_CHILDREN_WAITED_EXIT0" if all_normal else "INCOMPLETE_OR_FAILED_CHILD_PREFIX",
        "slot_count": 23, "source_files": C.WORKER_SOURCES, "worker_go_sha256": worker_go_sha,
        "controller_PID": os.getpid(), "environment_sha256": args.environment_sha256,
        "approved_environment_modules": environment["modules"],
        "slots": terminal_rows, "unknown_work": unknown, "failure": failure, "signals": stop.signals,
        "unknown_process_work": unknown_process, "unknown_partial_native_work": unknown_native,
        "unknown_partial_checkpoint_copy_work": unknown_copy,
        "controller_entry_through_children_seconds": time.perf_counter()-entered,
        "controller_resources_before_children_terminal_record": controller_resources(),
        "comparison_not_yet_run": True, "no_resume_or_retry": True}
    C.record(output/"terminal_children.json", terminal)
    if not all_normal:
        return 2
    comparison_started = time.perf_counter()
    try:
        resources_before_spawn(output, policy)  # reserve bounded comparison inputs/output too
        C.verify_bundle(registration)
        comparison_spec = importlib.util.spec_from_file_location("checkpoint23_independent_comparison", ROOT/"comparison.py")
        comparison = importlib.util.module_from_spec(comparison_spec)
        comparison_spec.loader.exec_module(comparison)
        report = comparison.evaluate(children, terminal, protocol, C.WORKER_SOURCES, worker_go_sha)
        C.record(output/"comparison.json", report)
        passed = (report.get("status") == "IMPLEMENTATION_AGREEMENT_PASS"
            and report.get("implementation_agreement_pass") is True and report.get("issues") == []
            and report.get("verified_worker_outputs") == 23
            and report.get("all23_externally_waited_exit0") is True)
    except BaseException as error:
        report = {"status": "COMPARISON_FAILED_OR_INTERRUPTED", "error": type(error).__name__+": "+str(error),
            "terminal_children_sha256": C.sha(output/"terminal_children.json"), "actual_child_waits": 23,
            "no_numerical_success_inferred": True}
        C.record(output/"comparison_failure.json", report)
        passed = False
    output_cap_failed = tree_bytes(output) > policy["total_output_cap_bytes"]
    if output_cap_failed:
        passed = False
    C.record(output/"controller_terminal.json", {"status": "COMPLETE_23_WAITED_AND_COMPARISON_PASS" if passed else "COMPARISON_FAILED",
        "terminal_children_sha256": C.sha(output/"terminal_children.json"),
        "comparison_sha256": C.sha(output/("comparison.json" if (output/"comparison.json").exists() else "comparison_failure.json")),
        "controller_entry_through_preterminal_record_seconds": time.perf_counter()-entered,
        "independent_comparison_seconds": time.perf_counter()-comparison_started,
        "actual_child_waits": 23, "controller_intended_returncode": 0 if passed else 2,
        "retained_output_cap_failed": output_cap_failed,
        "controller_resources_before_terminal_record": controller_resources(),
        "external_controller_wait_still_required": True, "cache_warmth_speed_claim": False})
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--registration-sha256")
    parser.add_argument("--root-go", type=Path)
    parser.add_argument("--root-go-sha256")
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--environment-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "HELD_CHECKPOINT23_NO_CHILD_NO_NUMERICAL_IMPORT",
            "slots": 23, "root_GO": "ABSENT_UNLESS_SEPARATELY_SUPPLIED", "retry": False,
            "native_runtime_preparation": "NOT_PERFORMED_OR_AUTHORIZED_BY_DEFAULT"}))
        return 0
    if any(getattr(args, key) is None for key in ("registration", "registration_sha256", "root_go",
        "root_go_sha256", "environment", "environment_sha256", "output")):
        raise ValueError("all exact registration/GO/environment/output arguments required")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
