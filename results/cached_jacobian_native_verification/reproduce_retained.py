"""Replay the frozen retained-array audit; no native solve or live observation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = "841b1ba8e583bad10a41ba49927fe6421bbf0249c077838469c3181c02094008"
SOURCE_SHA = "ffdc8e89b13fbaaf08239c4b5b4accf8b61c86c678eabac41091d72609edf1e1"
CRITERIA_SHA = "e18f1554cc99e2c71d9acfbb3a884fc5e4fa5d6608d1208adddac651902474c5"
PROTOCOL_SHA = "d1b6986d23d9840a1e66aa48a766a2cc624d2ee18a92f327d259067423fb8e85"
PRODUCTION_SHA = "e7ba5deef7a651f42d02b9f8e0bcbee9624291b8217ab378bdde7586c57cd4de"
EXTERNAL_SHA = "b7e1781b50cc0de7df3dc1685cd5e2a462904a8df2c7daea4d6b04f74666dfa4"
RESULT_SHA = "f92deb594263fbf83a13c25e3b23563d49740c011d202709175026b7019ba882"
REPORT_SHA = "1ef4baa3f0324b36c1178652d765e2845776c9349e0d2638adf86845b76e6a64"
ROUTE_SHA = "bf4c9d82c5af61e63b161c9a3073a783c0c028bc99123589c81a678caea1a0a6"
ROUTE_MANIFEST_SHA = "58bbd9646313954d46ebbc3676a007a1c1938c5b48d9bf4e85ab0341d9b42368"
COMMIT = "3b2f9b3ab024068eb618c699f41265bd37a89aa2"


def sha(path):
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_payload():
    path = HERE / "payload_manifest.json"
    if sha(path) != MANIFEST_SHA:
        raise ValueError("frozen payload manifest differs")
    manifest = read(path)
    if manifest["file_count"] != len(manifest["files"]) or manifest["file_count"] != 275:
        raise ValueError("frozen payload file denominator differs")
    for relative, row in manifest["files"].items():
        original = HERE / relative
        resolved = original.resolve()
        if (original.is_symlink() or not resolved.is_relative_to(HERE)
                or not resolved.is_file() or resolved.stat().st_size != row["bytes"]
                or sha(resolved) != row["sha256"]):
            raise ValueError("retained file differs: " + relative)
    actual = {p.relative_to(HERE).as_posix() for p in (HERE / "payload").rglob("*")
              if p.is_file()}
    expected = {name for name in manifest["files"] if name.startswith("payload/")}
    if actual != expected:
        raise ValueError("retained payload membership differs")
    return manifest


def same_scalars(expected, actual):
    if type(expected) is float:
        return (type(actual) is float and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=2e-12, abs_tol=2e-14))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(same_scalars(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(same_scalars(a, b) for a, b in zip(expected, actual, strict=True)))
    return type(expected) is type(actual) and expected == actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(HERE):
        raise ValueError("fresh output with an existing parent outside this archive required")
    verify_payload()
    source = HERE / "payload/audit/source_v2/audit_retained.py"
    packet = HERE / "payload/cohorts/v2_success/extracted"
    external = HERE / "payload/cohorts/v2_success/EXTERNAL_TERMINAL_OBSERVATION.json"
    reference_path = HERE / "payload/audit/actual_v2/evaluation/report.json"
    for path, expected in ((source, SOURCE_SHA), (source.with_name("criteria.json"), CRITERIA_SHA),
                           (source.with_name("native_protocol.json"), PROTOCOL_SHA),
                           (source.with_name("FROZEN_PRODUCTION_SOURCE.json"), PRODUCTION_SHA),
                           (external, EXTERNAL_SHA), (packet / "native/result.json", RESULT_SHA),
                           (reference_path, REPORT_SHA)):
        if sha(path) != expected:
            raise ValueError("frozen replay identity differs: " + path.name)
    output.mkdir()
    command = [sys.executable, "-B", "-I", str(source), "--execute-after-external-terminal",
               "--packet", str(packet), "--external-observation", str(external),
               "--output", str(output / "evaluation"), "--expected-source-sha256", SOURCE_SHA,
               "--expected-criteria-sha256", CRITERIA_SHA, "--expected-external-sha256", EXTERNAL_SHA,
               "--expected-result-sha256", RESULT_SHA, "--expected-route-source-sha256", ROUTE_SHA,
               "--expected-route-manifest-sha256", ROUTE_MANIFEST_SHA,
               "--expected-experiment-commit", COMMIT]
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    receipt = dict(status="FAILED_OR_INTERRUPTED_NO_VERIFIED_REPLAY", command=command,
                   fixed_payload_manifest_sha256=MANIFEST_SHA,
                   reused_historical_external_observation_sha256=EXTERNAL_SHA,
                   relocated_packet_directory=str(packet), native_executions=0, PDE_calls=0,
                   scope="Historical retained-array replay only. Reuses the unchanged saved external observation; no new GitHub observation, native wait, native solver, timing experiment or scientific benefit.")
    try:
        with (output / "audit.log").open("xb") as stream:
            child = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=120, check=False)
        receipt["actual_auditor_exit_code"] = child.returncode
        if child.returncode != 0:
            raise ValueError("frozen retained auditor failed; inspect audit.log")
        report = read(output / "evaluation/report.json")
        if (report["status"] != "RETAINED_ARRAY_SCALAR_AUDIT_CLEAR"
                or report["verified_credit"] is not True or report["issues"] != []
                or (report["checks"], report["numeric_comparisons"], report["npz_arrays_decoded"]) != (887, 108, 14)
                or report["mandatory_controls"] != 60 or report["ledger"]["actual_dispatches"] != 160):
            raise ValueError("required887/108/0/14 and60control/160dispatch replay gate differs")
        reference = read(reference_path)
        if report["ledger"] != reference["ledger"] or not same_scalars(reference["scalar_results"], report["scalar_results"]):
            raise ValueError("retained scalar or disjoint-work reproduction differs")
        verify_payload()
        receipt.update(status="REPRODUCED_FROZEN_RETAINED_FIELD_AUDIT", checks=887,
                       numeric_comparisons=108, issues=0, retained_arrays=14,
                       historical_controls=60, historical_dispatches=160,
                       actual_disjoint_physics_counts=report["ledger"]["disjoint_totals"],
                       reproduced_report_sha256=sha(output / "evaluation/report.json"),
                       scientific_goals_complete=False, student_final_design=False)
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "replay_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
