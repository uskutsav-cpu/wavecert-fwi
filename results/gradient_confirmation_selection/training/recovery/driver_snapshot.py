"""Audit/recover the frozen ENOSPC-interrupted gradient development study.

Default mode is read-only. --execute restarts only augmented_derivative using
the original frozen train_candidate function. No tracked source, protocol,
teacher archive, normalization, retained candidate, or original partial is
modified. Only atomic file writing receives a bounded ENOSPC retry/audit hook.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SHA256 = "5e56226791eb0d2297667bc3502f3d24813c81e27c4e4d3ed317b9f556871f98"
PROTOCOL_SHA256 = "9262ab19fc87c3d04f4bad1bcc2da5b6978babc10c55ee2e461c981c8fdefe82"
TEACHER_RECORD_SHA256 = "2140b9790845caae527ce6a0437a5d8563b377fda455a8d41f44ff0f1546657f"
LEGACY_RUNTIME_RECORD_SHA256 = "97efe4d72aa4166ef46f364b665c81f6d91ee9d5ca366c1f921a2634e5eadf5c"
RETAINED_WEIGHTS = {
    "raw_forward": "70e57af910e66088efae9bd3e6dd6afb504edcfc78996ea047a5c0c6ac79375c",
    "raw_derivative": "bb16f0c0dbc60a979f509d607b84932e0a8e7ade9b46576319e652e764e80b13",
    "augmented_forward": "aecf4af4f3b1c1b8da59697e280d85e45a7730008d51cef91568174c43f5a49c",
}
LABEL_SHA256 = {
    "train": "e0711c00a0b576c5ac7940e8a247d333031ca3d9091aada56e15bd0b8298a7d8",
    "validation": "3d969f7c71145faed10266ab99af30266aaa99994b5aa3814440b2a2979fc2c0",
    "validation_inverse_cases": "be429def325b741d415546167b8e13b5d52a3304d79f36efd3945a58363177db",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_training_rows(rows, count, candidate, protocol):
    training = protocol["training"]
    require(len(rows) == count, "optimizer history length does not match expected completed updates")
    for update, row in enumerate(rows, 1):
        require(int(row["optimizer_update"]) == update and int(row["batch_size"]) == training["batch_size"], "optimizer update sequence or batch size changed")
        require(float(row["derivative_weight"]) == candidate["derivative_weight"], "training derivative weight changed")
        require(row["jvp_training_evaluated"] == str(candidate["derivative_weight"] > 0), "training derivative-use flag changed")
        keys = ["total_loss", "full_field_relative_squared_l2", "receiver_relative_squared_l2", "preclip_parameter_gradient_norm", "learning_rate_used"]
        if candidate["derivative_weight"] > 0:
            keys.append("jvp_relative_squared_l2")
        require(all(math.isfinite(float(row[key])) for key in keys), "nonfinite retained training history")
        expected_lr = training["scheduler_eta_min"] + .5 * (training["learning_rate"] - training["scheduler_eta_min"]) * (1 + math.cos(math.pi * (update - 1) / training["optimizer_updates"]))
        require(math.isclose(float(row["learning_rate_used"]), expected_lr, rel_tol=1e-10, abs_tol=1e-12), "retained cosine learning-rate schedule changed")


def check_validation_rows(rows, protocol):
    expected = [0, *range(protocol["training"]["validation_interval"], protocol["training"]["optimizer_updates"] + 1, protocol["training"]["validation_interval"])]
    require([row["optimizer_update"] for row in rows] == expected, "retained validation schedule is incomplete")
    for row in rows:
        require(row["checkpoint_eligible"] == (row["optimizer_update"] > 0), "validation checkpoint eligibility changed")
        score = row["receiver_relative_squared_l2"] + row["jvp_relative_squared_l2"] + row["direction_loss"]
        require(math.isfinite(score) and math.isclose(score, row["selection_score"], rel_tol=1e-12, abs_tol=1e-12), "retained common selection score is inconsistent")
        require(math.isclose(row["direction_loss"], 1 - row["aggregate_gradient_cosine"], rel_tol=1e-12, abs_tol=1e-12), "retained inverse-gradient direction loss changed")


def audit(repository, runtime_reference):
    sys.path.insert(0, str(repository / "src"))
    import numpy as np
    import torch
    import wavecert.experiments.gradient_development as gd
    from wavecert.provenance import environment_record, sha256_file, source_digest

    require(Path(gd.__file__).resolve() == repository / "src/wavecert/experiments/gradient_development.py", "recovery imported the wrong training implementation")
    require(source_digest(repository) == SOURCE_SHA256, "main tracked source/config bytes differ from the frozen study")
    study = repository / "results/gradient_development"
    protocol_bytes = (study / "protocol.json").read_bytes()
    require(digest(protocol_bytes) == PROTOCOL_SHA256 and sha256_file(repository / "configs/gradient_development_protocol.json") == PROTOCOL_SHA256, "frozen protocol changed")
    protocol = json.loads(protocol_bytes)
    gd.validate_protocol(protocol)
    require(protocol["training"]["optimizer_updates"] == 2000 and protocol["training"]["batch_size"] == 32, "original optimizer budget changed")
    failure = read_json(study / "status.json")
    require(failure["status"] == "failed" and failure["protocol_sha256"] == PROTOCOL_SHA256 and "No space left on device" in failure["error"], "study is not the expected terminal ENOSPC failure")
    names = [row["name"] for row in protocol["candidates"]]
    require(names == [*RETAINED_WEIGHTS, "augmented_derivative"] and failure["completed_candidate_names"] == names[:3], "unexpected candidate completion state")
    require(sha256_file(study / "teacher_labels.json") == TEACHER_RECORD_SHA256, "teacher label record changed")
    records = read_json(study / "teacher_labels.json")
    labels = {}
    for role, expected in LABEL_SHA256.items():
        path = study / "labels" / f"{role}.npz"
        require(sha256_file(path) == expected == records[role]["artifact_sha256"], "saved teacher label archive changed")
        with np.load(path, allow_pickle=False) as archive:
            labels[role] = {key: archive[key].copy() for key in archive.files}
        for key, value in labels[role].items():
            require(gd._array_digest(value) == records[role]["array_sha256"][key], "saved teacher array content changed")
    _, current_inputs = gd.load_development_roles(protocol, repository)
    original_inputs = read_json(study / "input_audit.json")
    require(current_inputs == original_inputs, "original dataset identity/parent audit changed")
    raw_count = len(labels["train"]["models"]) // 2
    normalization = gd.WavefieldNormalization.from_arrays(labels["train"]["models"][:raw_count], labels["train"]["frequencies_hz"][:raw_count], labels["train"]["fields"][:raw_count])
    stored_normalization = read_json(study / "normalization.json")
    require({key: stored_normalization[key] for key in asdict(normalization)} == asdict(normalization), "raw-training-only normalization changed")
    retained, retained_file_hashes = [], {}
    for candidate in protocol["candidates"][:3]:
        name = candidate["name"]
        directory = study / "candidates" / name
        summary = read_json(directory / "summary.json")
        require(summary["status"] == "complete" and summary["name"] == name and summary["candidate"] == candidate, "retained candidate identity/status changed")
        require(summary["optimizer_updates_completed"] == 2000, "retained candidate did not finish 2000 updates")
        check_training_rows(csv_rows(directory / "training_history.csv"), 2000, candidate, protocol)
        validation = read_json(directory / "validation_history.json")
        check_validation_rows(validation, protocol)
        best = min(validation[1:], key=lambda row: (row["selection_score"], row["optimizer_update"]))
        require(summary["best_optimizer_update"] == best["optimizer_update"] and summary["best_validation_score"] == best["selection_score"], "retained validation-only checkpoint selection changed")
        require(summary["best_validation"] == best == read_json(directory / "best_validation.json"), "retained best-checkpoint validation receipt changed")
        checkpoint = directory / "best.pt"
        require(Path(summary["checkpoint_path"]).resolve() == checkpoint.resolve(), "retained checkpoint points elsewhere")
        require(sha256_file(checkpoint) == summary["checkpoint_sha256"] == RETAINED_WEIGHTS[name], "retained checkpoint weight bytes changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["development_source_sha256"] == SOURCE_SHA256 and payload["development_protocol_sha256"] == PROTOCOL_SHA256, "retained checkpoint source/protocol provenance differs")
        require(payload["best_optimizer_update"] == best["optimizer_update"] and payload["optimizer_updates_budget"] == 2000 and payload["candidate"] == candidate, "retained checkpoint budget/identity metadata changed")
        require(payload["normalization"] == asdict(normalization) and payload["model_config"] == protocol["model"], "retained model/normalization metadata changed")
        require(payload["teacher_label_artifacts"] == records, "retained checkpoint teacher archive provenance changed")
        require(payload["calibration_role_used_for_model_selection"] is False and payload["calibration_or_test_arrays_loaded"] is False and payload["external_test_arrays_loaded"] is False, "retained checkpoint reports forbidden held-out use")
        for path in directory.rglob("*"):
            if path.is_file():
                retained_file_hashes[str(path.relative_to(study))] = sha256_file(path)
        retained.append(summary)
    require(read_json(study / "completed_candidates.json") == retained, "three-candidate completion receipt differs from retained summaries")
    partial_directory = study / "candidates/augmented_derivative"
    require(not (partial_directory / "summary.json").exists(), "fourth candidate is already complete")
    partial_rows = csv_rows(partial_directory / "training_history.csv")
    check_training_rows(partial_rows, 100, protocol["candidates"][3], protocol)
    partial_validation = read_json(partial_directory / "validation_history.json")
    require(len(partial_validation) == 1 and partial_validation[0]["optimizer_update"] == 0 and partial_validation[0]["checkpoint_eligible"] is False, "fourth partial validation receipt differs from expected first-best save failure")
    torch.set_num_threads(protocol["training"]["torch_threads"])
    current_environment = environment_record(repository)
    runtime = {"original_start_runtime_receipt_available": False,
               "scope": "Current recovery environment reconstructed explicitly; original failed runner did not persist its full start environment record",
               "current_environment": current_environment}
    if runtime_reference.exists():
        require(sha256_file(runtime_reference) == LEGACY_RUNTIME_RECORD_SHA256, "contemporaneous runtime observation changed")
        observed = read_json(runtime_reference)
        comparison = observed["provenance"]
        for key in ("python", "platform", "machine"):
            require(current_environment[key] == comparison[key], f"recovery {key} differs from contemporaneous runtime observation")
        for package in ("numpy", "scipy", "torch"):
            require(current_environment["packages"][package] == comparison["packages"][package], f"recovery {package} differs from contemporaneous runtime observation")
        runtime["contemporaneous_observation"] = {"path": str(runtime_reference), "sha256": LEGACY_RUNTIME_RECORD_SHA256,
                                                   "recorded_utc": observed["recorded_utc"],
                                                   "matching_fields": ["python", "platform", "machine", "numpy", "scipy", "torch"],
                                                   "scope": "Concurrent legacy validation observation, not an original process-start receipt"}
    protected = {"protocol.json": PROTOCOL_SHA256, "teacher_labels.json": TEACHER_RECORD_SHA256,
                 "input_audit.json": sha256_file(study / "input_audit.json"), "normalization.json": sha256_file(study / "normalization.json"),
                 **{f"labels/{role}.npz": value for role, value in LABEL_SHA256.items()}, **retained_file_hashes}
    partial_files = {str(path.relative_to(partial_directory)): sha256_file(path) for path in partial_directory.rglob("*") if path.is_file()}
    return {"gd": gd, "torch": torch, "study": study, "protocol": protocol, "protocol_bytes": protocol_bytes,
            "records": records, "labels": labels, "normalization": normalization, "input_audit": original_inputs,
            "retained": retained, "protected_files": protected, "partial_directory": partial_directory,
            "partial_rows": partial_rows, "partial_validation": partial_validation, "partial_file_sha256": partial_files,
            "runtime": runtime, "failure": failure,
            "failed_status_file_mtime_utc": datetime.fromtimestamp((study / "status.json").stat().st_mtime, timezone.utc).isoformat()}


class RetryingAtomicWriter:
    """I/O-only intervention; no optimizer/model/loss code is replaced."""

    def __init__(self, original, candidate_output, partial_rows, initial_validation, reserve_path):
        self.original, self.candidate_output = original, candidate_output
        self.partial_rows, self.initial_validation = partial_rows, initial_validation
        self.reserve_path = reserve_path
        self.events, self.total_wait_seconds = [], 0.0
        self.prefix_verified = self.baseline_verified = False
        self.wait_budget_seconds, self.retry_interval_seconds = 900.0, 5.0
        self.reserve_released = False

    def __call__(self, path, data):
        path = Path(path)
        if path == self.candidate_output / "training_history.csv":
            rows = list(csv.DictReader(io.StringIO(data.decode())))
            if len(rows) >= len(self.partial_rows):
                require(rows[:len(self.partial_rows)] == self.partial_rows, "restarted first 100 optimizer records diverge from preserved original partial")
                self.prefix_verified = True
        if path == self.candidate_output / "validation_history.json":
            rows = json.loads(data)
            require(rows[0] == self.initial_validation[0], "restarted initial validation differs from preserved original baseline")
            self.baseline_verified = True
        while True:
            try:
                self.original(path, data)
                return
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                if self.reserve_path.exists():
                    self.reserve_path.unlink()
                    self.reserve_released = True
                event = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "path": str(path),
                         "bytes": len(data), "error": str(exc), "cumulative_wait_seconds": self.total_wait_seconds}
                self.events.append(event)
                print(json.dumps({"enospc_retry": event}), flush=True)
                if self.total_wait_seconds >= self.wait_budget_seconds:
                    raise OSError(errno.ENOSPC, "recovery atomic I/O retry budget exhausted after 15 minutes") from exc
                delay = min(self.retry_interval_seconds, self.wait_budget_seconds - self.total_wait_seconds)
                start = time.monotonic()
                time.sleep(delay)
                self.total_wait_seconds += time.monotonic() - start

    def receipt(self):
        return {"intervention": "Only module._atomic_bytes replaced with atomic-write ENOSPC retry and restart-prefix audit",
                "retry_interval_seconds": self.retry_interval_seconds, "cumulative_wait_budget_seconds": self.wait_budget_seconds,
                "observed_wait_seconds": self.total_wait_seconds, "events": self.events,
                "emergency_reserve_released_on_enospc": self.reserve_released,
                "first_100_optimizer_records_identical": self.prefix_verified,
                "initial_validation_baseline_identical": self.baseline_verified}


def execute(audit_data, repository, reserve_mib):
    gd, study = audit_data["gd"], audit_data["study"]
    started = time.monotonic()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    recovery = study / "recovery_attempts" / tag
    backup = study / "failure_attempts" / tag
    candidate_output = recovery / "candidate/augmented_derivative"
    recovery.mkdir(parents=True, exist_ok=False)
    backup.mkdir(parents=True, exist_ok=False)
    reserve = recovery / "enospc_emergency_reserve.bin"
    reserve_receipt = {"requested_bytes": reserve_mib * 1024 * 1024, "allocated_bytes": 0}
    if reserve_mib:
        try:
            with reserve.open("xb") as stream:
                block = os.urandom(1024 * 1024)
                for _ in range(reserve_mib):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            reserve_receipt["allocated_bytes"] = reserve.stat().st_size
        except OSError as exc:
            reserve.unlink(missing_ok=True)
            reserve_receipt["allocation_error"] = str(exc)
    original_atomic = gd._atomic_bytes
    writer = RetryingAtomicWriter(original_atomic, candidate_output, audit_data["partial_rows"], audit_data["partial_validation"], reserve)
    gd._atomic_bytes = writer
    driver_hash = digest(Path(__file__).read_bytes())
    recovery_receipt = {"schema_version": 1, "status": "running", "recovery_id": tag,
                        "driver_path": str(Path(__file__).resolve()), "driver_sha256": driver_hash,
                        "frozen_source_sha256": SOURCE_SHA256, "protocol_sha256": PROTOCOL_SHA256,
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                        "runtime_audit": audit_data["runtime"], "original_failure": audit_data["failure"],
                        "failed_status_file_mtime_utc": audit_data["failed_status_file_mtime_utc"],
                        "preserved_partial_sha256": audit_data["partial_file_sha256"],
                        "protected_file_sha256": audit_data["protected_files"], "reserve": reserve_receipt,
                        "scope": "Restart fourth candidate only from its identical original seed and 2000-update budget; keep all original files and partial attempt"}
    try:
        for name in ("status.json", "completed_candidates.json"):
            writer(backup / name, (study / name).read_bytes())
        for name in audit_data["partial_file_sha256"]:
            writer(backup / "partial_candidate/augmented_derivative" / name, (audit_data["partial_directory"] / name).read_bytes())
        gd._write_json(backup / "preservation_receipt.json", recovery_receipt)
        gd._write_json(recovery / "recovery_receipt.json", recovery_receipt)
        gd._write_json(study / "status.json", {"status": "recovering-fourth-candidate", "protocol_sha256": PROTOCOL_SHA256, "recovery_id": tag})
        require(shutil.disk_usage(study).free >= 32 * 1024 * 1024, "less than 32 MiB available before recovery; no training launched")
        result = gd.train_candidate(
            audit_data["protocol"]["candidates"][3], audit_data["labels"]["train"], audit_data["labels"]["validation"],
            audit_data["labels"]["validation_inverse_cases"], audit_data["normalization"], audit_data["protocol"], candidate_output,
            protocol_sha256=PROTOCOL_SHA256, input_audit=audit_data["input_audit"], label_records=audit_data["records"], source_sha256=SOURCE_SHA256,
        )
        require(writer.prefix_verified and writer.baseline_verified, "recovery did not verify preserved restart prefix/baseline")
        check_training_rows(csv_rows(candidate_output / "training_history.csv"), 2000, audit_data["protocol"]["candidates"][3], audit_data["protocol"])
        check_validation_rows(read_json(candidate_output / "validation_history.json"), audit_data["protocol"])
        from wavecert.provenance import sha256_file, source_digest
        require(source_digest(repository) == SOURCE_SHA256, "frozen source changed during recovery")
        for name, expected in audit_data["protected_files"].items():
            require(sha256_file(study / name) == expected, f"protected original input/candidate file changed: {name}")
        for name, expected in audit_data["partial_file_sha256"].items():
            require(sha256_file(audit_data["partial_directory"] / name) == expected, "original failed partial was changed")
        candidates = [*audit_data["retained"], result]
        selection = gd.select_winner(candidates, [row["name"] for row in audit_data["protocol"]["candidates"]])
        selection["source_and_inputs_unchanged"] = True
        recovery_receipt.update({"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                                 "recovery_driver_elapsed_seconds": time.monotonic() - started, "io_policy": writer.receipt(),
                                 "fourth_candidate_restarted_from_seed": True, "fourth_failed_attempt_updates_preserved": 100,
                                 "fourth_successful_attempt_updates": 2000, "fourth_total_executed_updates_across_attempts": 2100,
                                 "retained_three_candidates_retrained": False})
        summary = {
            "schema_version": 1, "status": "complete", "recovered_from_interruption": True,
            "protocol_sha256": PROTOCOL_SHA256, "elapsed_seconds": None,
            "elapsed_seconds_scope": "No single uninterrupted run: original failed-attempt total wall time was not persisted. Keep original completed-candidate and recovery timing receipts separately; do not sum these as a complete end-to-end benchmark.",
            "input_audit": audit_data["input_audit"], "normalization": asdict(audit_data["normalization"]),
            "teacher_labels": audit_data["records"], "candidates": candidates, "selection": selection,
            "provenance": audit_data["runtime"]["current_environment"],
            "provenance_scope": "Recovery-time reconstruction on matching frozen source, not a fabricated original start-runtime receipt",
            "original_start_runtime_receipt_available": False,
            "initial_source_sha256_recovered_from_retained_checkpoints": SOURCE_SHA256,
            "source_sha256_at_end": SOURCE_SHA256,
            "source_unchanged_during_run": None,
            "inputs_unchanged_during_run": None,
            "source_unchanged_during_recovery": True,
            "recorded_initial_end_and_recovery_source_digests_match": True,
            "inputs_unchanged_during_recovery": True, "recovery": recovery_receipt,
            "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False,
            "limitations": ["Interrupted study; original full start-runtime/end-to-end wall-time receipt unavailable.",
                            "Fourth candidate reran its original 2000-update budget; 100 failed-attempt updates remain additional spent work.",
                            "Atomic I/O retries may add elapsed wall time; no loss/model/optimizer/protocol changes were made.",
                            "Development-only validation on four parents; no independent confirmation or deployment authorization."]}
        gd._write_json(study / "completed_candidates.json", candidates)
        gd._write_json(study / "selection.json", selection)
        gd._write_json(study / "summary.json", summary)
        gd._write_json(recovery / "recovery_receipt.json", recovery_receipt)
        gd._write_json(study / "status.json", {"status": "complete", "protocol_sha256": PROTOCOL_SHA256, "recovery_id": tag, "recovered_from_enospc": True})
        print(json.dumps({"status": "complete", "recovery_id": tag, "selection": selection}), flush=True)
    except BaseException as exc:
        recovery_receipt.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                 "finished_utc": datetime.now(timezone.utc).isoformat(), "io_policy": writer.receipt()})
        if reserve.exists():
            reserve.unlink()
        print(json.dumps({"recovery_failed_receipt": recovery_receipt}), flush=True)
        # After an exhausted ENOSPC budget do not silently skip the receipt;
        # try once with any released reserve and leave the complete stdout copy.
        for path in (recovery / "recovery_receipt.json", study / "status.json"):
            try:
                original_atomic(path, (json.dumps(recovery_receipt, indent=2, allow_nan=False) + "\n").encode())
            except OSError as receipt_error:
                print(json.dumps({"failure_receipt_write_error": str(receipt_error), "path": str(path)}), flush=True)
        raise
    finally:
        gd._atomic_bytes = original_atomic
        reserve.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually rerun the fourth candidate after review; default is read-only audit")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parent / "wavecert-fwi")
    parser.add_argument("--runtime-reference", type=Path, default=Path(__file__).resolve().parent / "legacy-common-validation.json")
    parser.add_argument("--emergency-reserve-mib", type=int, default=8)
    args = parser.parse_args()
    require(args.emergency_reserve_mib in {0, 8}, "only zero or eight MiB task-owned emergency reserve is permitted")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    repository = args.repository.resolve()
    data = audit(repository, args.runtime_reference.resolve())
    print(json.dumps({"status": "recovery-audit-passed; execution not started" if not args.execute else "recovery-audit-passed",
                      "source_sha256": SOURCE_SHA256, "protocol_sha256": PROTOCOL_SHA256,
                      "retained_complete_candidates": [row["name"] for row in data["retained"]],
                      "retained_update_counts": [row["optimizer_updates_completed"] for row in data["retained"]],
                      "partial_fourth_updates": len(data["partial_rows"]),
                      "original_start_runtime_receipt_available": False,
                      "runtime_observation_match": "contemporaneous_observation" in data["runtime"],
                      "tracked_source_or_protocol_writes": False,
                      "io_policy": "Atomic I/O only; ENOSPC every5s, cumulative900s budget, optional8MiB owned reserve"}), flush=True)
    if args.execute:
        execute(data, repository, args.emergency_reserve_mib)


if __name__ == "__main__":
    main()
