"""Future isolated import-identity shim; numerical imports require admission."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import resource
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot23_common_entry", ROOT/"pilot23_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--environment-sha256")
    parser.add_argument("--slot-dir", type=Path)
    parser.add_argument("--worker-go", type=Path)
    parser.add_argument("--worker-go-sha256")
    parser.add_argument("--phase")
    parser.add_argument("--slot-id")
    parser.add_argument("--checkpoints", type=int)
    parser.add_argument("--direction")
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--sign", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("HELD_IMPORT_IDENTITY_SHIM_NO_IMPORT_NO_OPERATOR")
        return 0
    entered = time.perf_counter()
    protocol = C.verify_worker()
    if C.sha(args.environment) != args.environment_sha256 or C.sha(args.worker_go) != args.worker_go_sha256:
        raise ValueError("explicit exact environment/workerGO bytes required")
    receipt = C.read_json(args.environment)
    C.verify_environment(receipt, protocol)
    go = C.read_json(args.worker_go)
    if (go.get("authorization") != "GO_CONSTANT_MEDIUM_CHECKPOINT_PILOT"
            or go.get("execution_authorized") is not True or go.get("source_files") != C.WORKER_SOURCES
            or go.get("independent_source_review_clear") is not True
            or go.get("criteria_frozen_before_execution") is not True):
        raise ValueError("exact reviewed native worker GO required")
    slot = C.canonical(args.slot_dir)
    policy = C.read_json(ROOT/"controller_policy.json")
    resource.setrlimit(resource.RLIMIT_FSIZE, (policy["per_file_hard_limit_bytes"], policy["per_file_hard_limit_bytes"]))
    imports = {}
    for name, expected in receipt["modules"].items():
        module = importlib.import_module(name)  # future authorized worker only
        origin = Path(module.__file__).resolve()
        if str(origin) != expected["path"] or C.sha(origin) != expected["sha256"]:
            raise ValueError("actual imported module origin differs: "+name)
        imports[name] = {"path": str(origin), "sha256": C.sha(origin)}
    devito, revolve = sys.modules["devito"], sys.modules["pyrevolve"]
    if (devito.DevitoCheckpoint.__module__ != "devito.checkpointing.checkpoint"
            or devito.CheckpointOperator.__module__ != "devito.checkpointing.checkpoint"
            or revolve.Revolver is not sys.modules["pyrevolve.pyrevolve"].MemoryRevolver):
        raise ValueError("Noop or unregistered checkpoint/scheduler class resolved")
    C.record(slot/"import_identity.json", {"status": "SELECTED_ACTUAL_IMPORT_ORIGINS_MATCH_APPROVED_ENVIRONMENT",
        "slot_id": args.slot_id, "worker_output": str(slot/"worker"),
        "PID": os.getpid(), "PPID": os.getppid(), "process_group_ID": os.getpgrp(),
        "session_ID": os.getsid(0), "selected_modules": imports,
        "entry_through_import_gate_seconds": time.perf_counter()-entered,
        "environment_sha256": args.environment_sha256, "worker_go_sha256": args.worker_go_sha256,
        "controlled_native_environment": {key: os.environ.get(key) for key in C.NATIVE_ENV},
        "native_PDE_calls_before_worker": 0,
        "scope": "Selected actual source/module origins only; no whole-runtime attestation."})
    command = [str(ROOT/"worker/run_checkpoint_pilot.py"), "--execute", "--root-go", str(args.worker_go),
               "--output", str(slot/"worker"), "--phase", args.phase]
    for key, value in (("--checkpoints", args.checkpoints), ("--direction", args.direction),
                       ("--epsilon", args.epsilon), ("--sign", args.sign)):
        if value is not None:
            command.extend([key, str(value)])
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
