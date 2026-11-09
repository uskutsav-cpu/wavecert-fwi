"""Held single-slot real-case worker. Default: stdlib metadata, no numerical imports.

No supervisor/auditor/GO is supplied. A separate exact root GO and reviewed
future parent are required even for the case0 two-return integration pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contract():
    return {name: sha(HERE/name) for name in ("realcase_worker.py", "case_runtime.py", "project_protocol.json",
        "FROZEN_PRODUCTION_SOURCE.json", "api/STAGED_FILE_MANIFEST.json", "api/protocol.json",
        "api/FROZEN48_SCHEDULE.json", "inputs/frozen_initials.npz",
        "inputs/FROZEN_INITIAL_INPUT_RECEIPT.json", "inputs/RETAINED_INPUTS_01.json")}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_gate(repo, p):
    manifest = HERE/"api/STAGED_FILE_MANIFEST.json"
    if sha(manifest) != p["historical_API"]["manifest_sha256"]:
        raise ValueError("historical API manifest changed")
    for relative, row in json.loads(manifest.read_bytes())["files"].items():
        original = HERE/"api"/relative
        path = original.resolve()
        if (not path.is_relative_to(HERE/"api") or original.is_symlink()
                or sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]):
            raise ValueError("historical API source/control evidence changed: " + relative)
    for relative, expected in p["inputs_sha256"].items():
        path = (repo/relative).resolve()
        if not path.is_relative_to(repo) or sha(path) != expected:
            raise ValueError("raw input changed: " + relative)
    for key in ("initializer_input",):
        row = p[key]
        if sha(HERE/row["path"]) != row["sha256"] or sha(HERE/row["receipt_path"]) != row["receipt_sha256"]:
            raise ValueError("exact retained initializer input changed")
    if (sha(HERE/p["retained_input_schema_path"]) != p["retained_input_schema_sha256"]
            or sha(HERE/p["schedule_path"]) != p["schedule_sha256"]
            or sha(HERE/"api/protocol.json") != p["historical_API"]["protocol_sha256"]
            or sha(HERE/"FROZEN_PRODUCTION_SOURCE.json") != p["production_manifest_sha256"]):
        raise ValueError("new project input/protocol dependencies changed")
    frozen = json.loads((HERE/"FROZEN_PRODUCTION_SOURCE.json").read_bytes())
    if frozen["source_sha256"] != p["production_source_sha256"] or len(frozen["files"]) != 161:
        raise ValueError("frozen9f production contract changed")
    for relative, expected in frozen["files"].items():
        if sha(repo/relative) != expected:
            raise ValueError("production9f source changed: " + relative)
    historical = json.loads((HERE/"api/protocol.json").read_bytes())
    for key in ("physics", "frequencies_hz", "observation_blocks", "real_residual_terms", "arms",
                "squared_slowness_bounds", "optimizer_deadline_seconds", "complete_return_cap_seconds"):
        if p[key] != historical[key]:
            raise ValueError("project amendment changed frozen numerical setting: " + key)
    if ([c["identity"] for c in p["cases"]] != historical["identities"]
            or p["allowed_mode"] != "case0_two_return_integration_pilot"
            or p["allowed_slot_indices"] != [0,1] or p["full48_execution_available"] is not False):
        raise ValueError("project case scope/pilot admission changed")
    return {"production_source_sha256": frozen["source_sha256"], "production_files": 161,
        "historical_API_files": len(json.loads(manifest.read_bytes())["files"]), "source_files": contract()}


def admit_values(go, request, p, source_files, *, now, parent_pid):
    flags = ("execution_authorized", "independent_worker_source_review_clear",
        "external_supervisor_review_clear", "independent_auditor_review_clear", "runtime_review_clear")
    if (go.get("authorization") != p["root_go_authorization"] or any(go.get(k) is not True for k in flags)
            or go.get("mode") != "case0_two_return_integration_pilot"
            or go.get("allowed_slot_indices") != [0, 1] or go.get("full48_authorized") is not False
            or go.get("source_files") != source_files
            or go.get("production_source_sha256") != p["production_source_sha256"]):
        raise ValueError("separate exact root GO/reviews required; no full48 admission")
    for key in ("supervisor_source_sha256", "auditor_source_sha256", "audit_criteria_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(go.get(key))) is None:
            raise ValueError("future reviewed supervisor/auditor/criteria must be bound")
    index = request.get("slot_index")
    started = request.get("absolute_started")
    if (type(index) is not int or index not in [0, 1] or type(started) not in (int, float)
            or not math.isfinite(started) or not 0 <= started <= now
            or request.get("parent_pid") != parent_pid or type(request.get("parent_pid")) is not int
            or request.get("clock") != "time.perf_counter_same_host"
            or request.get("optimizer_absolute_deadline") != started+7.5
            or request.get("complete_return_absolute_deadline") != started+8.
            or request.get("source_contract_sha256") != hashlib.sha256(canonical(source_files)).hexdigest()):
        raise ValueError("actual earlier-origin/parent/slot/source request differs")
    rows = json.loads((HERE/p["schedule_path"]).read_bytes())
    slot = rows[index]
    if (slot["case_index"] != 0 or slot["repetition"] != 0 or slot["status"] != "unattempted"
            or request.get("arm") != slot["arm"] or request.get("case_index") != slot["case_index"]):
        raise ValueError("pilot request differs from fixed first two slots")
    if now-started >= 7.5:
        raise TimeoutError("absolute optimizer deadline elapsed before imports")
    return {"slot": slot, "absolute_started": float(started), "pre_admission_age": now-started,
        "source_files": source_files, "mode": "case0_two_return_integration_pilot",
        "external_wait_and_audit_required": True}


def admit(args, p, *, clock=time.perf_counter):
    go = json.loads(args.root_go.read_bytes())
    request = json.loads(args.request.read_bytes())
    source = source_gate(args.repo.resolve(), p)
    result = admit_values(go, request, p, source["source_files"], now=clock(), parent_pid=os.getppid())
    result.update(root_go_sha256=sha(args.root_go), request_sha256=sha(args.request), source=source)
    return result


def limits(p):
    if platform.system() != p["runtime"]["platform"] or platform.python_version() != p["runtime"]["python"]:
        raise ValueError("exact Linux/Python runtime required")
    for name in ("numpy", "scipy"):
        if importlib.metadata.version(name) != p["runtime"][name]:
            raise ValueError("exact pinned package runtime required")
    if any(os.environ.get(k) != value for k, value in p["threads"].items()):
        raise ValueError("thread policy must be set before imports")
    for kind, value in ((resource.RLIMIT_AS, p["resources"]["address_space_limit_bytes"]),
                        (resource.RLIMIT_CPU, p["resources"]["CPU_limit_seconds"])):
        resource.setrlimit(kind, (value, value))
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(p["resources"]["worker_emergency_wall_seconds"])


def plain(value):
    """Failure evidence: keep every branch, represent arrays by identity, not zeros."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "__array_interface__"):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype),
            "array_payload_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": repr(value)}
    if hasattr(value, "item"):
        return plain(value.item())
    return {"unserializable_type": type(value).__name__}


class Gate:
    def __init__(self, output, p, *, clock=time.perf_counter):
        self.output, self.p, self.clock = output, p, clock
        self.base, self.project_protocol_sha = HERE, sha(HERE/"project_protocol.json")
        self.project_source_check = None
        self.attempt, self.context, self.stop_signals = {}, {}, []
        self.sequence = 0

    def event(self, kind, **values):
        row = {"sequence": self.sequence, "kind": kind, "observed_perf_counter": self.clock(), **plain(values)}
        raw = canonical(row)+b"\n"
        if self.output_bytes()+len(raw) > self.p["resources"]["output_cap_bytes"]-self.p["resources"]["failure_receipt_reserve_bytes"]:
            raise OSError("fixed output/failure reserve exhausted")
        with (self.output/"events.jsonl").open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1

    def output_bytes(self):
        return sum(q.stat().st_size for q in self.output.iterdir() if q.is_file())

    def evidence(self):
        return {"attempt": plain(self.attempt), "context": plain(self.context),
            "stop_signals": list(self.stop_signals), "unknown_tail_if_dispatch_interrupted": True}


def require_same_case_binding(reference, current):
    keys = {"case_index", "raw24_m_sha256", "initial49_m_sha256", "truth49_m_sha256",
        "all8_observations_sha256", "survey_sha256"}
    if set(reference) != keys or set(current) != keys or reference != current:
        raise ValueError("new arm/repetition prepared-case bytes differ; no replacement or old bitguard")


def lifecycle(args, p, *, admission=admit, limiter=limits, runner=None, clock=time.perf_counter):
    output = args.output.resolve()
    output.mkdir()  # no overwrite/retry/resume
    gate = Gate(output, p, clock=clock)
    write_new(output/"project_plan.json", p)
    write_new(output/"source_contract_before.json", contract())
    previous = {}
    try:
        accepted = admission(args, p, clock=clock)
        write_new(output/"admission.json", accepted)
        gate.context["admission"] = accepted
        gate.project_source_check = lambda: source_gate(args.repo.resolve(),p)
        gate.event("exact_GO_source_request_persisted_before_numerical_imports")
        limiter(p)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.getsignal(sig)
            def stop(signum, frame):
                gate.stop_signals.append(signal.Signals(signum).name)
                # Repeats are nonraising during failure evidence persistence.
                if len(gate.stop_signals) == 1:
                    raise KeyboardInterrupt("worker stop request")
            signal.signal(sig, stop)
        if runner is None:
            name = "_trf_held_case_runtime"
            if name in sys.modules:
                raise ValueError("runtime import name occupied")
            spec = importlib.util.spec_from_file_location(name, HERE/"case_runtime.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            runner = module.run_case
        result = runner(args, p, accepted, gate)
        if gate.stop_signals:
            raise InterruptedError("stop request prevents successful worker credit")
        if source_gate(args.repo.resolve(),p)["source_files"] != accepted["source_files"]:
            raise ValueError("worker/source/input files changed during run")
        result.update(status="WORKER_RETURNED_PENDING_EXTERNAL_WAIT_AND_INDEPENDENT_AUDIT",
            source_files=contract(), evidence=gate.evidence(), observed_self_peak_RSS_bytes=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1024 if platform.system()=="Linux" else 1),
            RSS_scope="Worker selfpeak observed before result serialization; no complete-OS or parent peak claim.",
            actual_external_wait_required=True, scientific_goals_complete=False)
        write_new(output/"result.json", result)
        if gate.output_bytes() > p["resources"]["output_cap_bytes"]:
            raise OSError("fixed output cap exceeded including result")
        return 0
    except BaseException as exc:  # noqa: BLE001 -- retain every available branch; no retry or fake tail count
        # Deferral prevents repeated parent stop signals interrupting our own
        # failure receipt. No worker child exists here; parent owns OS cleanup.
        for sig in previous:
            signal.signal(sig, lambda signum, frame: gate.stop_signals.append(signal.Signals(signum).name))
        write_new(output/"failure.json", {"status": "FAILED_OR_INTERRUPTED_NO_RETRY", "exception_type": type(exc).__name__,
            "message": str(exc), "evidence": gate.evidence(), "source_files": contract(),
            "no_complete_process_eligibility_claim": True, "scientific_goals_complete": False})
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("repo", "root-go", "request", "output"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "HELD_SOURCE_ONLY_REALCASE_WORKER", "source_files": contract(),
            "project_execution_authorized": False, "numerical_imports": 0, "PDE_calls": 0,
            "supervisor_or_independent_auditor_supplied": False}))
        return 0
    if any(getattr(args, name) is None for name in ("repo", "root_go", "request", "output")):
        parser.error("explicit repo/root-go/request/new output required")
    p = json.loads((HERE/"project_protocol.json").read_bytes())
    return lifecycle(args, p)


if __name__ == "__main__":
    raise SystemExit(main())
