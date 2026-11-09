"""Stdlib source/admission utilities for the separately versioned23 pilot."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
WORKER_SOURCES = {
    "run_checkpoint_pilot.py": "31969bd3e072dcbf554e0cd98f022b0e35dd5438ef12566f9c23fa98eee76e9b",
    "pilot_accounting.py": "1b0094af99fb57e44966e18808aa80f939a2a83771ed98c53a94a5e1460f33f2",
    "protocol.json": "08f262f638da9a72b262d252ba8e3eaab4452d772d8c80606bc44e7137fba866",
    "RUNTIME_SOURCE_PINS.json": "bfbc2e31cd4ff7ae2a0f0342dc3633c5e4dc0189daea0d06eaeb12de1040350f",
}
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
NATIVE_ENV = {"DEVITO_LANGUAGE": "C", "DEVITO_MPI": "0", "DEVITO_AUTOTUNING": "off",
    "DEVITO_AUTOPADDING": "0", "DEVITO_SAFE_MATH": "1", "DEVITO_JIT_BACKDOOR": "0",
    "DEVITO_FIRST_TOUCH": "0", "MPLBACKEND": "Agg"}


def sha(path, cap=64*1024**2):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("bounded regular source/evidence file required: "+str(path))
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path, cap=1024**2):
    if path.is_symlink() or path.stat().st_size > cap:
        raise ValueError("bounded non-symlink JSON required")
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def record(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def canonical(path):
    value = Path(path)
    if not value.is_absolute() or value != value.resolve():
        raise ValueError("canonical absolute path required: "+str(path))
    return value


def verify_worker():
    for name, expected in WORKER_SOURCES.items():
        if sha(ROOT/"worker"/name) != expected:
            raise ValueError("frozen worker bytes differ: "+name)
    return read_json(ROOT/"worker/protocol.json")


def expected_slots(protocol):
    rows = []
    def add(slot_id, phase, checkpoints=None, direction=None, epsilon=None, epsilon_index=None, sign=None):
        rows.append({"slot_index": len(rows), "slot_id": slot_id, "phase": phase,
            "checkpoints": checkpoints, "direction": direction, "epsilon": epsilon,
            "epsilon_index": epsilon_index, "sign": sign, "state": "unattempted",
            "actual_returncode": None, "numerical_eligible": False})
    add("saved", "saved")
    add("checkpoint2", "checkpoint2", checkpoints=2)
    add("checkpoint4", "checkpoint4", checkpoints=4)
    for direction in protocol["constant_control"]["directions"]:
        for index, epsilon in enumerate(protocol["finite_difference"]["epsilon"]):
            for sign in protocol["finite_difference"]["signs"]:
                add(f"fd_{direction}_{index}_{sign}", "fd", direction=direction,
                    epsilon=epsilon, epsilon_index=index, sign=sign)
    return rows


def verify_plan(protocol):
    plan = read_json(ROOT/"FROZEN_PLAN23.json")
    if (plan.get("schema") != "checkpoint_pilot23_plan_v1" or plan.get("slot_count") != 23
            or plan.get("slots") != expected_slots(protocol)
            or plan.get("worker_source_files") != WORKER_SOURCES):
        raise ValueError("complete unchanged23-slot constant-medium plan required")
    return plan


def verify_bundle(registration):
    manifest_path = ROOT/"BUNDLE_MANIFEST.json"
    if sha(manifest_path) != registration.get("bundle_manifest_sha256"):
        raise ValueError("exact registered controller bundle required")
    manifest = read_json(manifest_path)
    required = {"control_pilot23.py", "pilot23_common.py", "worker_entry.py", "comparison.py",
        "controller_policy.json", "FROZEN_PLAN23.json", *("worker/"+name for name in WORKER_SOURCES)}
    if manifest.get("schema") != "checkpoint_pilot23_bundle_v1" or set(manifest.get("files", {})) != required:
        raise ValueError("exact10-file runtime bundle denominator required")
    for name, expected in manifest["files"].items():
        path = ROOT/name
        if not path.resolve().is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError("controller bundle source differs: "+name)
    return manifest


def module_name(path):
    value = path.removesuffix(".py").replace("/", ".")
    return value.removesuffix(".__init__")


def verify_environment(receipt, protocol):
    """No numerical imports: metadata and selected approved bytes only."""
    runtime = protocol["runtime"]
    if (receipt.get("schema") != "checkpoint_pilot_ephemeral_linux_environment_v1"
            or receipt.get("ephemeral") is not True or receipt.get("native_use_review_clear") is not True
            or platform.system() != receipt.get("system") or receipt.get("system") != "Linux"
            or platform.machine() != receipt.get("machine") or receipt.get("machine") != "x86_64"
            or platform.python_version() != runtime["python"] or receipt.get("python_version") != runtime["python"]):
        raise ValueError("separately reviewed exact ephemeral Linux runtime required")
    prefix = canonical(receipt["prefix"])
    invocation = Path(sys.executable).absolute()
    if (str(invocation) != receipt.get("python_executable")
            or str(invocation.resolve()) != receipt.get("python_realpath")
            or Path(sys.prefix).resolve() != prefix or not invocation.is_relative_to(prefix)):
        raise ValueError("controller must run under the exact reviewed ephemeral interpreter")
    if sha(invocation.resolve()) != receipt.get("python_binary_sha256"):
        raise ValueError("approved interpreter binary bytes differ")
    versions = {name: importlib.metadata.version(name) for name in ("devito", "pyrevolve", "numpy", "scipy")}
    if versions != receipt.get("packages") or any(versions[key] != runtime[key] for key in versions):
        raise ValueError("registered runtime package versions differ")
    pins = read_json(ROOT/"worker/RUNTIME_SOURCE_PINS.json")
    expected = {module_name(row["path"]): row for row in pins["files"]}
    required = set(expected) | {"pyrevolve", "pyrevolve.crevolve", "numpy", "scipy"}
    modules = receipt.get("modules", {})
    if set(modules) != required:
        raise ValueError("exact selected source/package/scheduler module identity map required")
    for name, row in modules.items():
        if set(row) != {"path", "sha256"}:
            raise ValueError("selected module identity must contain exactly path/hash: "+name)
        path = canonical(row["path"])
        if not path.is_relative_to(prefix) or sha(path) != row["sha256"]:
            raise ValueError("approved runtime module bytes differ: "+name)
        if name in expected:
            primary = expected[name]
            installed = Path(importlib.metadata.distribution(primary["distribution"]).locate_file(primary["path"])).resolve()
            if path != installed or row["sha256"] != primary["sha256"]:
                raise ValueError("selected installed primary source differs: "+name)
    return {"python_executable": str(invocation), "packages": versions,
        "selected_module_count": len(modules), "selected_modules": modules,
        "identity_scope": "Selected13 primary Python APIs plus package roots and compiled scheduler only; not whole-distribution/binary attestation."}


def controlled_environment():
    value = dict(os.environ)
    for key in list(value):
        if key.startswith("DEVITO_"):
            value.pop(key)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
                "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        value.pop(key, None)
    value.update(THREADS)
    value.update(NATIVE_ENV)
    value.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")
    return value
