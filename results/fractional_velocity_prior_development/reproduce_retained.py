"""Reproduce the retained scalar audit after verifying the frozen archive; no PDE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SOURCE_SHA = "57946398ebd6d3eaad35ea73491ee50f3bf00c64ef42d6067a1a2fd522c040a4"
TRACE_SHA = "f2eaf67a296aa00e22e21d8ce410d72f715b317097da4f2b486fc3b2ff6235cc"
REPORT_SHA = "c3b528d942cba432b38be6bd2f46139402b058687e9e220781e10d57a4161baa"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, repo, output = Path(__file__).resolve().parent, args.repo.resolve(), args.output.resolve()
    if (output.exists() or not output.parent.is_dir()
            or output.is_relative_to(archive) or output.is_relative_to(repo)):
        raise ValueError("new output with an existing parent outside the archive/repository required")
    manifest = json.loads((archive / "archive_manifest.json").read_bytes())
    for relative, expected in manifest["files"].items():
        path = archive / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(archive)
                or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]):
            raise ValueError("archive content differs: " + relative)
    retained = archive / "cohort_01/extracted"
    run = retained / "study"
    child = json.loads((retained / "supervision/child_exit.json").read_bytes())
    original = archive / "cohort_01/ROOT_TERMINAL_OBSERVATION.json"
    terminal = json.loads(original.read_bytes())
    if (type(child["actual_returncode"]) is not int or child["actual_returncode"] != 0
            or terminal["actual_exit_code"] != child["actual_returncode"]
            or terminal["status_sha256"] != sha(run / "status.json")
            or terminal["complete_receipt_sha256"] != sha(run / "complete_receipt.json")):
        raise ValueError("retained terminal execution evidence differs")
    output.mkdir()
    terminal.update(run_directory=str(run), relocation_of_retained_root_observation_sha256=sha(original),
                    scope="Relocation of retained root terminal evidence; no new live process/GitHub observation or PDE.")
    observation = output / "relocated_terminal_observation.json"
    observation.write_text(json.dumps(terminal, indent=2) + "\n")
    source = archive / "portable_controls/fractional_velocity_prior48_post_audit/audit_retained.py"
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    command = [sys.executable, "-B", "-I", str(source), "--execute-after-root-terminal",
               "--repo", str(repo), "--run", str(run), "--output", str(output / "audit"),
               "--terminal-observation", str(observation), "--expected-source-sha256", SOURCE_SHA,
               "--expected-trace-sha256", TRACE_SHA, "--expected-terminal-sha256", sha(observation)]
    with (output / "audit.log").open("xb") as stream:
        child_result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    receipt = dict(actual_child_exit=child_result.returncode, command=command,
                   retained_manifest_sha256=sha(archive / "archive_manifest.json"),
                   scope="Retained scalar reproduction only; no new physics, timing run or independent geology.")
    if child_result.returncode == 0:
        saved = archive / "retained_audit/report.json"
        assert sha(saved) == REPORT_SHA
        reference = json.loads(saved.read_bytes())
        reproduced = json.loads((output / "audit/report.json").read_bytes())
        assert reproduced["verified_full_study_credit"] is True and not reproduced["issues"]
        assert reproduced["summary"]["all48_eligible"] is True
        assert reproduced["summary"]["proposed_development_continuation_gate_pass"] is False
        for field in ("cases", "case_arm_groups"):
            assert len(reference["summary"][field]) == len(reproduced["summary"][field])
            for expected, actual in zip(reference["summary"][field], reproduced["summary"][field], strict=True):
                assert expected.keys() == actual.keys()
                for key, value in expected.items():
                    assert (math.isclose(value, actual[key], rel_tol=2e-12, abs_tol=2e-14)
                            if type(value) is float else value == actual[key]), (field, key)
        receipt.update(status="REPRODUCED_NEGATIVE_DEVELOPMENT_RESULT", checks=reproduced["checks"],
                       numeric_comparisons=reproduced["numeric_comparisons"], issues=0,
                       all48_eligible=True, successful_cases=0, required_cases=6,
                       continuation_gate_pass=False, scientific_goals_complete=False)
    else:
        receipt["status"] = "AUDITOR_FAILED_INSPECT_RETAINED_LOG"
    (output / "reproduction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return child_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
