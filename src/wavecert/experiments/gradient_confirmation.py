"""Prospective model-selection freeze and fresh-index confirmation controls.

This is an assistant-development workflow, not student preregistration. It
never chooses a model from calibration/test/confirmation outcomes. Original
training, calibration and external-stress artifacts remain read-only inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.experiments.openfwi_stress import prepare_external_models
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.statistics.finite_population import finite_population_failure_summary
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate

FROZEN_PLAN_SHA256 = "8330f33ee38255dbb14f54e2ddcd6f154bbe7d5f4a40b11f2560b652f563eb41"
RECOVERY_DRIVER_SHA256 = "f38f3ed712346a198e364e92585adf1f04dc167f4b8eaa835311715bfe5298bc"
RECOVERY_TEACHER_SHA256 = "2140b9790845caae527ce6a0437a5d8563b377fda455a8d41f44ff0f1546657f"
RECOVERY_LABEL_SHA256 = {
    "train": "e0711c00a0b576c5ac7940e8a247d333031ca3d9091aada56e15bd0b8298a7d8",
    "validation": "3d969f7c71145faed10266ab99af30266aaa99994b5aa3814440b2a2979fc2c0",
    "validation_inverse_cases": "be429def325b741d415546167b8e13b5d52a3304d79f36efd3945a58363177db",
}


def _json(path):
    return json.loads(Path(path).read_text())


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    _require(not Path(relative).is_absolute() and path.is_relative_to(root) and path != root,
             "artifact paths must be contained relative paths")
    return path


def _new_directory(path):
    path = Path(path).resolve()
    _require(not path.exists(), "output must be a new directory; immutable prior artifacts are preserved")
    path.mkdir(parents=True)
    return path


def _write_once(root, relative, data):
    path = _inside(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _runtime_contract(provenance):
    """Bind only numerical runtime versions and the planned CPU device."""
    return {"device": "cpu", "python": provenance["python"],
            "packages": {name: provenance["packages"][name] for name in ("numpy", "scipy", "torch")}}


def verify_prior_stage_binding(previous_provenance, expected_runtime, repository_root):
    """Reject edited recorded sources/runtime; report unused new source files.

    Source files added after a freeze may be unrelated analysis tools. They
    cannot replace any recorded file or be imported into this pipeline without
    a new immutable freeze attempt. This is a reproducibility guard, not proof
    that arbitrary dynamic code elsewhere never influenced a process.
    """
    root = Path(repository_root).resolve()
    manifest = previous_provenance["source_files"]
    _require(bool(manifest), "previous stage source manifest is empty; create a new immutable freeze")
    digest = hashlib.sha256()
    for name, checksum in sorted(manifest.items()):
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
        _require(sha256_file(_inside(root, name)) == checksum,
                 f"previously frozen source changed: {name}; create a new immutable freeze attempt")
    _require(digest.hexdigest() == previous_provenance["source_sha256"],
             "previous stage source manifest/hash mismatch")
    current = environment_record(root)
    _require(expected_runtime == _runtime_contract(previous_provenance) == _runtime_contract(current),
             "Python/numpy/scipy/torch runtime or CPU device contract changed; create a new immutable freeze attempt")
    new_files = sorted(set(current["source_files"]) - set(manifest))
    imported = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                if getattr(module, "__file__", None)}
    _require(not any(_inside(root, name) in imported for name in new_files),
             "a newly added source file is imported into the pipeline; create a new immutable freeze attempt")
    return {"all_recorded_source_files_unchanged": True, "numerical_runtime_versions_unchanged": True,
            "runtime_contract": expected_runtime, "new_unimported_source_files": new_files,
            "scope": "New files are permitted only as unused/unrelated additions; no recorded source/config file can change. Rest of the environment is recorded without exact optional-package equality."}


def confirmation_indices(excluded, *, seed, family_code):
    """Uniform original-index sampling from the fixed remaining 400 slots."""
    _require(len(excluded) == len(set(excluded)) == 100 and
             all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 500 for i in excluded),
             "old selection must contain 100 distinct original indices")
    population = sorted(set(range(500)) - set(excluded))
    selected = sorted(np.random.default_rng(np.random.SeedSequence([seed, family_code])).choice(
        population, size=100, replace=False,
    ).tolist())
    return population, selected


def validate_confirmation_plan(plan_path, repository_root):
    """Check published receipts and prospective folds without model predictions."""
    root, plan_path = Path(repository_root), Path(plan_path)
    _require(sha256_file(plan_path) == FROZEN_PLAN_SHA256, "frozen prospective plan/fold bytes changed")
    plan = _json(plan_path)
    _require(plan["schema_version"] == 1 and plan["stage"] == "A-prospective-plan-before-winner-known",
             "unsupported confirmation plan")
    old = plan["old_external"]
    protocol_path = _inside(root, old["protocol_path"])
    receipt_path = _inside(root, old["published_preparation_manifest_path"])
    _require(sha256_file(protocol_path) == old["protocol_sha256"] and
             sha256_file(receipt_path) == old["published_preparation_manifest_sha256"],
             "published historical protocol or preparation receipt changed")
    protocol, receipt = _json(protocol_path), _json(receipt_path)
    _require(receipt["protocol_sha256"] == old["protocol_sha256"], "historical receipt/protocol mismatch")
    _require(plan["geometry"] == protocol["geometry"] and plan["preprocessing"] == protocol["preprocessing"],
             "confirmation acquisition or preprocessing changed")
    _require(plan["selection"]["seed"] == 2026110731 and
             len(plan["families"]) == 2 and {f["name"] for f in plan["families"]} == {f["name"] for f in receipt["families"]},
             "confirmation families or sample seed changed")
    for family in plan["families"]:
        previous = next(f for f in receipt["families"] if f["name"] == family["name"])
        original = next(f for f in protocol["families"] if f["name"] == family["name"])
        for key in ("family_code", "member", "expected_crc32", "expected_shape", "expected_dtype"):
            _require(family[key] == original[key], f"pinned raw member identity changed: {key}")
        _require(family["raw_npy_sha256"] == previous["sha256"] and
                 family["raw_npy_relative_path"] == previous["path"] and
                 family["excluded_old_indices"] == previous["selected_indices"],
                 "raw payload identity or old excluded cohort changed")
        population, selected = confirmation_indices(previous["selected_indices"],
            seed=plan["selection"]["seed"], family_code=family["family_code"])
        _require(family["analysis_population_size"] == 400 and family["confirmation_sample_size"] == 100 and
                 family["eligible_complement_indices"] == population and
                 family["confirmation_selected_indices"] == selected and
                 family["eligible_complement_indices_sha256"] == _sha(json.dumps(population, separators=(",", ":")).encode()),
                 "fresh confirmation fold or finite analysis population changed")
    expected_labels = {"legacy_forward", "legacy_jvp"}
    _require(len(plan["legacy_models"]) == 2 and {c["label"] for c in plan["legacy_models"]} == expected_labels,
             "both legacy model controls are required")
    for model in plan["legacy_models"]:
        _require(sha256_file(_inside(root, model["checkpoint_path"])) == model["checkpoint_sha256"] and
                 sha256_file(_inside(root, model["calibration_snapshot_path"])) == model["calibration_summary_sha256"],
                 "frozen legacy checkpoint/calibration changed")
    training = plan["development_training"]
    _require(sha256_file(_inside(root, training["protocol_path"])) == training["protocol_sha256"],
             "prospective training protocol changed")
    return plan


def _verify_training_source(provenance, source_root, expected_end):
    """Verify recorded training bytes; later added tools are not training edits."""
    digest = hashlib.sha256()
    manifest = provenance["source_files"]
    _require(bool(manifest), "training source manifest is empty")
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", provenance["git_revision"]],
                                    cwd=source_root, text=True).splitlines()
    expected = {name for name in names if _training_source_path(name)}
    _require(set(manifest) == expected, "training source manifest is incomplete for the declared commit")
    for name, checksum in sorted(manifest.items()):
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
        current = _inside(source_root, name)
        historical = subprocess.check_output(
            ["git", "show", f"{provenance['git_revision']}:{name}"], cwd=source_root,
        )
        _require(sha256_file(current) == checksum == _sha(historical),
                 f"training source bytes no longer match the declared frozen commit: {name}")
    _require(digest.hexdigest() == provenance["source_sha256"] == expected_end,
             "training start/end source identity mismatch")


def _training_source_path(name):
    path = Path(name)
    return (name == "pyproject.toml" or (name.startswith("src/") and path.suffix == ".py") or
            name.startswith("configs/") or (name.startswith("scripts/") and path.suffix in {".py", ".sh"}) or
            (path.parent.as_posix() == ".github/workflows" and path.suffix == ".yml") or
            (path.parent.as_posix() == "." and path.match("requirements*.txt")))


def _recovery_driver_path(training_dir, receipt):
    """Prefer a portable adjacent archive without rewriting the old receipt."""
    archived = _inside(training_dir, f"recovery_attempts/{receipt['recovery_id']}/recovery_driver.py")
    path = archived if archived.exists() else Path(receipt["driver_path"])
    _require(receipt["driver_sha256"] == RECOVERY_DRIVER_SHA256 and
             sha256_file(path) == RECOVERY_DRIVER_SHA256, "reviewed recovery driver bytes changed")
    return path


def _candidate_directory(training_dir, name, admission):
    relative = (f"recovery_attempts/{admission['recovery_id']}/candidate/{name}"
                if admission["mode"] == "explicit-audited-ENOSPC-restart" and name == "augmented_derivative"
                else f"candidates/{name}")
    return _inside(training_dir, relative)


def _check_recorded_checkpoint_path(recorded, training_dir, directory):
    """Read known contained files; historical absolute strings may relocate."""
    expected = _inside(training_dir, (directory / "best.pt").relative_to(training_dir).as_posix())
    relative = expected.relative_to(Path(training_dir).resolve())
    recorded = Path(recorded)
    if recorded.is_absolute():
        # Preserve the original receipt, never access its former absolute path.
        _require(recorded.parts[-len(relative.parts):] == relative.parts,
                 "recorded checkpoint path does not identify this canonical/recovery candidate")
    else:
        _require(_inside(training_dir, recorded) == expected,
                 "recorded checkpoint path does not identify this canonical/recovery candidate")


def _recovered_training_admission(summary, status, training_dir, protocol):
    """Admit the specifically reviewed I/O-only restart with explicit limits.

    This does not turn unknown original runtime or continuous source monitoring
    into True. Stored initial/end/recovery identities and restart equality are
    evidence of reproducibility, not a reconstructed uninterrupted experiment.
    """
    import torch

    from wavecert.experiments.gradient_development import WavefieldNormalization

    _require(summary.get("recovered_from_interruption") is True and
             summary["source_unchanged_during_run"] is None and summary["inputs_unchanged_during_run"] is None and
             summary["source_unchanged_during_recovery"] is True and
             summary["inputs_unchanged_during_recovery"] is True and
             summary["recorded_initial_end_and_recovery_source_digests_match"] is True and
             summary["original_start_runtime_receipt_available"] is False and summary["elapsed_seconds"] is None,
             "recovered training must preserve unknown original runtime/whole-run monitoring")
    receipt = summary["recovery"]
    tag = receipt["recovery_id"]
    _require(isinstance(tag, str) and Path(tag).name == tag and tag not in {"", ".", ".."},
             "invalid recovery attempt identity")
    recovery = _inside(training_dir, f"recovery_attempts/{tag}")
    failure = _inside(training_dir, f"failure_attempts/{tag}")
    _require(_json(recovery / "recovery_receipt.json") == receipt and receipt["status"] == "complete" and
             status.get("recovered_from_enospc") is True and status["recovery_id"] == tag,
             "recovery completion receipt is missing or inconsistent")
    _recovery_driver_path(training_dir, receipt)
    _require(receipt["frozen_source_sha256"] == summary["source_sha256_at_end"] ==
             summary["initial_source_sha256_recovered_from_retained_checkpoints"] == summary["provenance"]["source_sha256"] and
             receipt["protocol_sha256"] == summary["protocol_sha256"],
             "initial/end/recovery source or protocol identities differ")
    _require(receipt["runtime_audit"]["original_start_runtime_receipt_available"] is False and
             receipt["runtime_audit"]["current_environment"] == summary["provenance"],
             "recovery runtime reconstruction differs from the recorded environment")
    _require(receipt["fourth_candidate_restarted_from_seed"] is True and
             receipt["fourth_failed_attempt_updates_preserved"] == 100 and
             receipt["fourth_successful_attempt_updates"] == 2000 and
             receipt["fourth_total_executed_updates_across_attempts"] == 2100 and
             receipt["retained_three_candidates_retrained"] is False and
             receipt["io_policy"]["first_100_optimizer_records_identical"] is True and
             receipt["io_policy"]["initial_validation_baseline_identical"] is True,
             "recovery must retain its original 100 extra updates and fixed restart budget")
    original = _json(failure / "status.json")
    _require(original == receipt["original_failure"] and original["status"] == "failed" and
             "No space left on device" in original["error"] and
             original["completed_candidate_names"] == [c["name"] for c in protocol["candidates"][:3]] and
             original["protocol_sha256"] == summary["protocol_sha256"],
             "preserved original ENOSPC failure is inconsistent")
    preservation = _json(failure / "preservation_receipt.json")
    _require(preservation["status"] == "running" and
             all(receipt.get(key) == value for key, value in preservation.items() if key != "status"),
             "initial recovery preservation receipt changed")
    _require(_json(failure / "completed_candidates.json") == summary["candidates"][:3],
             "recovery must preserve all three previously completed candidates")
    required_protected = {"protocol.json", "teacher_labels.json", "input_audit.json", "normalization.json"}
    required_protected |= {f"labels/{role}.npz" for role in RECOVERY_LABEL_SHA256}
    required_protected |= {f"candidates/{c['name']}/{name}" for c in protocol["candidates"][:3]
                          for name in ("summary.json", "training_history.csv", "validation_history.json", "best.pt", "best_validation.json")}
    _require(required_protected <= set(receipt["protected_file_sha256"]),
             "recovery receipt omits protected original inputs/candidate files")
    for name, checksum in receipt["protected_file_sha256"].items():
        _require(sha256_file(_inside(training_dir, name)) == checksum, f"protected recovery input changed: {name}")
    partial = _inside(training_dir, "candidates/augmented_derivative")
    partial_copy = failure / "partial_candidate/augmented_derivative"
    _require({"training_history.csv", "validation_history.json"} <= set(receipt["preserved_partial_sha256"]),
             "preserved restart baseline/history is missing")
    for name, checksum in receipt["preserved_partial_sha256"].items():
        _require(sha256_file(_inside(partial, name)) == sha256_file(_inside(partial_copy, name)) == checksum,
                 "preserved original failed partial changed")
    restarted = recovery / "candidate/augmented_derivative"
    def rows(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    prefix = rows(partial / "training_history.csv")
    baseline = _json(partial / "validation_history.json")
    _require(len(prefix) == 100 and rows(restarted / "training_history.csv")[:100] == prefix,
             "restarted first 100 optimizer records differ from original partial")
    _require(len(baseline) == 1 and baseline[0]["optimizer_update"] == 0 and
             baseline[0]["checkpoint_eligible"] is False and
             _json(restarted / "validation_history.json")[0] == baseline[0],
             "restarted initial validation differs from the original baseline")
    teacher_path = training_dir / "teacher_labels.json"
    _require(sha256_file(teacher_path) == RECOVERY_TEACHER_SHA256,
             "original pinned teacher-label receipt changed")
    records = _json(teacher_path)
    _require(summary["teacher_labels"] == records and _json(training_dir / "input_audit.json") == summary["input_audit"],
             "recovered summary teacher/input receipts differ")
    for role, checksum in RECOVERY_LABEL_SHA256.items():
        _require(sha256_file(training_dir / f"labels/{role}.npz") == checksum == records[role]["artifact_sha256"],
                 "original pinned teacher archive changed")
    with np.load(training_dir / "labels/train.npz", allow_pickle=False) as archive:
        raw_count = len(archive["models"]) // 2
        normalization = vars(WavefieldNormalization.from_arrays(archive["models"][:raw_count],
            archive["frequencies_hz"][:raw_count], archive["fields"][:raw_count]))
    stored = _json(training_dir / "normalization.json")
    _require(summary["normalization"] == normalization == {key: stored[key] for key in normalization},
             "shared raw-training-only normalization changed")
    for candidate, item in zip(protocol["candidates"], summary["candidates"], strict=True):
        directory = restarted if candidate["name"] == "augmented_derivative" else training_dir / "candidates" / candidate["name"]
        history = rows(directory / "training_history.csv")
        _require(len(history) == 2000, "recovered candidate optimizer history is incomplete")
        for update, row in enumerate(history, 1):
            keys = ["total_loss", "full_field_relative_squared_l2", "receiver_relative_squared_l2",
                    "preclip_parameter_gradient_norm", "learning_rate_used"]
            if candidate["derivative_weight"] > 0:
                keys.append("jvp_relative_squared_l2")
            lr = protocol["training"]["scheduler_eta_min"] + .5 * (
                protocol["training"]["learning_rate"] - protocol["training"]["scheduler_eta_min"]) * (
                    1 + math.cos(math.pi * (update - 1) / 2000))
            _require(int(row["optimizer_update"]) == update and int(row["batch_size"]) == 32 and
                     float(row["derivative_weight"]) == candidate["derivative_weight"] and
                     row["jvp_training_evaluated"] == str(candidate["derivative_weight"] > 0) and
                     all(math.isfinite(float(row[key])) for key in keys) and
                     math.isclose(float(row["learning_rate_used"]), lr, rel_tol=1e-10, abs_tol=1e-12),
                     "recovered optimizer history has nonfinite losses or changed training budget/schedule")
        best = _json(directory / "best_validation.json")
        _require(item["candidate"] == candidate and item["best_validation"] == best and
                 best in _json(directory / "validation_history.json"),
                 "recovered best-checkpoint validation receipt changed")
        _check_recorded_checkpoint_path(item["checkpoint_path"], training_dir, directory)
        payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        _require(payload["development_source_sha256"] == receipt["frozen_source_sha256"] and
                 payload["development_protocol_sha256"] == receipt["protocol_sha256"] and
                 payload["candidate"] == candidate and payload["optimizer_updates_budget"] == 2000 and
                 payload["best_optimizer_update"] == item["best_optimizer_update"] and
                 payload["best_validation_selection_score"] == item["best_validation_score"] and
                 payload["normalization"] == normalization and payload["model_config"] == protocol["model"] and
                 payload["teacher_label_artifacts"] == records and
                 payload["dataset_manifest_sha256"] == summary["input_audit"]["dataset_manifest_sha256"] and
                 payload["training_manifest_sha256"] == summary["input_audit"]["role_manifest_sha256"] and
                 payload["calibration_role_used_for_model_selection"] is False and
                 payload["calibration_or_test_arrays_loaded"] is False and payload["external_test_arrays_loaded"] is False,
                 "recovered checkpoint metadata differs from the frozen source/protocol/input budget")
    return {"mode": "explicit-audited-ENOSPC-restart", "recovery_id": tag,
        "whole_run_source_and_inputs_unchanged": None, "original_start_runtime_receipt_available": False,
        "recorded_initial_end_recovery_source_identities_match": True, "source_and_inputs_unchanged_during_recovery": True,
        "logical_optimizer_updates_per_candidate": 2000, "fourth_candidate_actual_executed_updates": 2100,
        "total_actual_executed_updates": 8100, "driver_sha256": RECOVERY_DRIVER_SHA256,
        "receipt_sha256": sha256_file(recovery / "recovery_receipt.json"),
        "limitations": summary["limitations"],
        "scope": "Explicit restart admission, not uninterrupted-run provenance; original runtime and total wall time remain unknown."}


def _training_selection(plan, training_dir, source_root):
    """Audit all candidates and the existing selector, without held-out loads."""
    training_dir, source_root = Path(training_dir).resolve(), Path(source_root).resolve()
    summary, status = _json(training_dir / "summary.json"), _json(training_dir / "status.json")
    selection, completed = _json(training_dir / "selection.json"), _json(training_dir / "completed_candidates.json")
    cfg = plan["development_training"]
    _require(status["status"] == summary["status"] == "complete" and selection["source_and_inputs_unchanged"] is True,
             "training must be complete with its frozen selection receipt")
    _require(summary["calibration_or_test_arrays_loaded"] is False and summary["external_test_arrays_loaded"] is False and
             summary["input_audit"]["opened_array_roles"] == ["train", "validation"],
             "development training must exclude calibration/test/external arrays")
    _require(summary["protocol_sha256"] == status["protocol_sha256"] == cfg["protocol_sha256"] ==
             sha256_file(training_dir / "protocol.json"), "training protocol differs from the prospective plan")
    _verify_training_source(summary["provenance"], source_root, summary["source_sha256_at_end"])
    _require(summary["provenance"]["git_revision"].startswith(cfg["source_commit"]),
             "training source commit differs from the prospective plan")
    protocol = _json(training_dir / "protocol.json")
    if summary.get("recovered_from_interruption") is True:
        admission = _recovered_training_admission(summary, status, training_dir, protocol)
    else:
        _require(summary["source_unchanged_during_run"] is True and summary["inputs_unchanged_during_run"] is True,
                 "training must be complete with unchanged source and inputs or explicit audited recovery")
        admission = {"mode": "uninterrupted-recorded-run", "whole_run_source_and_inputs_unchanged": True}
    names = cfg["required_candidates"]
    _require([c["name"] for c in protocol["candidates"]] == names and
             protocol["training"]["optimizer_updates"] == cfg["required_optimizer_updates_per_candidate"] == 2000,
             "all four frozen candidate definitions and update budgets are required")
    _require(summary["candidates"] == completed and len(completed) == 4 and
             [c["name"] for c in completed] == names,
             "all four completed candidates must remain in the training report")
    for relative, checksum in summary["input_audit"]["input_file_sha256"].items():
        _require(sha256_file(_inside(source_root, relative)) == checksum, "training input bytes changed")
    dataset = _inside(source_root, plan["selected_calibration"]["dataset_path"])
    _require(sha256_file(dataset / "dataset_manifest.json") == plan["selected_calibration"]["dataset_manifest_sha256"] and
             sha256_file(dataset / "split_manifest.csv") == plan["selected_calibration"]["role_manifest_sha256"],
             "source role/dataset identity changed after training")
    snapshots = {}
    for item in completed:
        candidate_dir = _candidate_directory(training_dir, item["name"], admission)
        _check_recorded_checkpoint_path(item["checkpoint_path"], training_dir, candidate_dir)
        _require(item["status"] == "complete" and item["optimizer_updates_completed"] == 2000 and
                 item["calibration_or_test_arrays_loaded"] is False and item["external_test_arrays_loaded"] is False and
                 _json(candidate_dir / "summary.json") == item, "a candidate is incomplete or its report changed")
        with (candidate_dir / "training_history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        _require([int(row["optimizer_update"]) for row in history] == list(range(1, 2001)) and
                 all(int(row["batch_size"]) == protocol["training"]["batch_size"] for row in history),
                 "candidate optimizer-update history is incomplete")
        validations = _json(candidate_dir / "validation_history.json")
        interval = protocol["training"]["validation_interval"]
        _require([r["optimizer_update"] for r in validations] == list(range(0, 2001, interval)),
                 "candidate validation history is incomplete")
        eligible = [r for r in validations if r["optimizer_update"] > 0]
        _require(all(r["checkpoint_eligible"] is (r["optimizer_update"] > 0) for r in validations),
                 "initial or missing checkpoint eligibility changed")
        for row in validations:
            score = row["receiver_relative_squared_l2"] + row["jvp_relative_squared_l2"] + row["direction_loss"]
            _require(np.isfinite(score) and math.isclose(score, row["selection_score"], rel_tol=1e-10) and
                     math.isclose(row["direction_loss"], 1 - row["aggregate_gradient_cosine"], rel_tol=1e-10, abs_tol=1e-12),
                     "candidate validation score components disagree")
        best = min(eligible, key=lambda r: (r["selection_score"], r["optimizer_update"]))
        _require(best["optimizer_update"] == item["best_optimizer_update"] and
                 best["selection_score"] == item["best_validation_score"], "candidate checkpoint selection changed")
        weight = candidate_dir / "best.pt"
        _require(sha256_file(weight) == item["checkpoint_sha256"], "a completed candidate checkpoint changed")
        snapshots[item["name"]] = {"path": weight, "sha256": item["checkpoint_sha256"],
                                    "best_optimizer_update": item["best_optimizer_update"]}
    chosen = min(completed, key=lambda r: (r["best_validation_score"], names.index(r["name"])))
    _require(selection["winner"] == chosen["name"] and selection["checkpoint_sha256"] == chosen["checkpoint_sha256"] and
             selection["best_optimizer_update"] == chosen["best_optimizer_update"] and
             selection["validation_selection_score"] == chosen["best_validation_score"],
             "winner differs from the frozen validation-only selection rule")
    return summary, selection, snapshots, admission


def freeze_selected_checkpoint(*, plan_path, repository_root, training_dir, training_source_root, output_dir):
    """Stage B: write an immutable selection freeze before new held-out calls."""
    root = Path(repository_root)
    plan = validate_confirmation_plan(plan_path, root)
    summary, selection, candidates, admission = _training_selection(plan, training_dir, training_source_root)
    output = _new_directory(output_dir)
    _write_once(output, "plan.json", Path(plan_path).read_bytes())
    for name in ("summary.json", "selection.json", "status.json", "protocol.json", "input_audit.json"):
        _write_once(output, f"training/{name}", (Path(training_dir) / name).read_bytes())
    if admission["mode"] == "explicit-audited-ENOSPC-restart":
        training = Path(training_dir).resolve()
        tag = admission["recovery_id"]
        receipt = summary["recovery"]
        _write_once(output, "training/recovery/driver_snapshot.py", _recovery_driver_path(training, receipt).read_bytes())
        paths = {f"recovery_attempts/{tag}/recovery_receipt.json", "teacher_labels.json", "normalization.json"}
        paths |= {p.relative_to(training).as_posix() for p in (training / f"failure_attempts/{tag}").rglob("*") if p.is_file()}
        for item in candidates.values():
            paths |= {p.relative_to(training).as_posix() for p in item["path"].parent.iterdir()
                      if p.is_file() and p.suffix in {".json", ".csv"}}
        for name in sorted(paths):
            _write_once(output, f"training/recovery/evidence/{name}", _inside(training, name).read_bytes())
        _write_once(output, "training/recovery/admission.json", _json_bytes(admission))
    models = []
    for old in plan["legacy_models"]:
        checkpoint_path, calibration_path = f"checkpoints/{old['label']}.pt", f"calibration/{old['label']}.json"
        _write_once(output, checkpoint_path, _inside(root, old["checkpoint_path"]).read_bytes())
        _write_once(output, calibration_path, _inside(root, old["calibration_snapshot_path"]).read_bytes())
        models.append({"label": old["label"], "checkpoint_path": checkpoint_path,
                       "checkpoint_sha256": old["checkpoint_sha256"], "calibration_path": calibration_path,
                       "calibration_sha256": old["calibration_summary_sha256"]})
    winner = candidates[selection["winner"]]
    data = winner["path"].read_bytes()
    _require(_sha(data) == winner["sha256"], "selected weights changed while freezing")
    _write_once(output, "checkpoints/development_winner.pt", data)
    models.append({"label": "development_winner", "checkpoint_path": "checkpoints/development_winner.pt",
                   "checkpoint_sha256": winner["sha256"], "candidate_name": selection["winner"]})
    files = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    stage_environment = environment_record(root)
    frozen = {"schema_version": 1, "stage": "B-selection-frozen-before-new-calibration",
        "created_utc": _utc(), "plan_sha256": sha256_file(plan_path), "files": files, "models": models,
        "selected_candidate": selection["winner"], "selection_json_sha256": sha256_file(Path(training_dir) / "selection.json"),
        "all_candidate_checkpoints": {name: {k: v for k, v in item.items() if k != "path"} for name, item in candidates.items()},
        "training_source_commit": summary["provenance"]["git_revision"],
        "training_source_sha256": summary["provenance"]["source_sha256"],
        "training_source_and_inputs_unchanged": admission["whole_run_source_and_inputs_unchanged"],
        "training_admission": admission,
        "new_selected_checkpoint_calibration_or_test_performed_by_this_stage": False,
        "scope": "Immutable assistant-development selection/weight receipt; not student preregistration or unseen-data proof.",
        "runtime_contract": _runtime_contract(stage_environment), "provenance": stage_environment}
    _write_once(output, "freeze.json", _json_bytes(frozen))
    return frozen


def validate_selection_freeze(freeze_dir):
    root = Path(freeze_dir)
    frozen = _json(root / "freeze.json")
    _require(frozen["stage"] == "B-selection-frozen-before-new-calibration" and
             {m["label"] for m in frozen["models"]} == {"legacy_forward", "legacy_jvp", "development_winner"} and
             len(frozen["models"]) == 3, "a complete three-model selection freeze is required before evaluation")
    required = {"plan.json", "training/summary.json", "training/selection.json", "training/status.json",
                "training/protocol.json", "training/input_audit.json"}
    _require(required <= set(frozen["files"]) and
             frozen["new_selected_checkpoint_calibration_or_test_performed_by_this_stage"] is False,
             "selection freeze lacks its completed-training receipts")
    for relative, checksum in frozen["files"].items():
        _require(sha256_file(_inside(root, relative)) == checksum, "immutable selection-freeze input changed")
    if frozen.get("training_admission", {}).get("mode") == "explicit-audited-ENOSPC-restart":
        admission = frozen["training_admission"]
        training_summary = _json(root / "training/summary.json")
        receipt_path = _inside(root, f"training/recovery/evidence/recovery_attempts/{admission['recovery_id']}/recovery_receipt.json")
        _require(frozen["training_source_and_inputs_unchanged"] is None and
                 _json(root / "training/recovery/admission.json") == admission and
                 training_summary["source_unchanged_during_run"] is None and training_summary["inputs_unchanged_during_run"] is None and
                 _json(receipt_path) == training_summary["recovery"] and
                 sha256_file(receipt_path) == admission["receipt_sha256"] and
                 admission["fourth_candidate_actual_executed_updates"] == 2100 and
                 admission["total_actual_executed_updates"] == 8100 and
                 admission["original_start_runtime_receipt_available"] is False and
                 admission["whole_run_source_and_inputs_unchanged"] is None and
                 sha256_file(root / "training/recovery/driver_snapshot.py") == admission["driver_sha256"] == RECOVERY_DRIVER_SHA256,
                 "recovered selection freeze must retain interruption and restart evidence")
    else:
        _require(frozen["training_source_and_inputs_unchanged"] is True,
                 "selection freeze lacks unchanged training or explicit audited recovery")
    _require(sha256_file(root / "plan.json") == frozen["plan_sha256"], "selection-freeze plan changed")
    _require(frozen["plan_sha256"] == FROZEN_PLAN_SHA256, "selection freeze uses another prospective plan")
    plan = _json(root / "plan.json")
    _require(set(frozen["all_candidate_checkpoints"]) == set(plan["development_training"]["required_candidates"]) and
             frozen["selected_candidate"] in frozen["all_candidate_checkpoints"],
             "selection freeze must retain all four candidate checkpoint identities")
    for model in frozen["models"]:
        _require(sha256_file(_inside(root, model["checkpoint_path"])) == model["checkpoint_sha256"],
                 "frozen model weights changed")
    return frozen


def _run_selected_calibration(*, freeze_dir, repository_root, dataset_dir, output_dir):
    """Stage C: new selected-model calibration plus explicitly reused ID/OOD."""
    frozen = validate_selection_freeze(freeze_dir)
    prior_binding = verify_prior_stage_binding(frozen["provenance"], frozen["runtime_contract"], repository_root)
    from wavecert.experiments.real_wavecert import run_real_wavecert_study

    plan = _json(Path(freeze_dir) / "plan.json")
    config = plan["selected_calibration"]
    _require(sha256_file(Path(dataset_dir) / "dataset_manifest.json") == config["dataset_manifest_sha256"] and
             sha256_file(Path(dataset_dir) / "split_manifest.csv") == config["role_manifest_sha256"],
             "selected calibration dataset differs from the prospective plan")
    output = _new_directory(output_dir)
    winner = next(m for m in frozen["models"] if m["label"] == "development_winner")
    checkpoint = _inside(freeze_dir, winner["checkpoint_path"])
    source = environment_record(repository_root)
    _write_once(output, "started.json", _json_bytes({"status": "running", "created_utc": _utc(),
        "selection_freeze_sha256": sha256_file(Path(freeze_dir) / "freeze.json"), "provenance": source}))
    report = run_real_wavecert_study(dataset_dir=dataset_dir, checkpoint_path=checkpoint,
        output_dir=output / "selected_calibration", alpha=config["alpha"],
        calibration_cases=config["calibration_slice_limit"], evaluation_cases=config["evaluation_slice_limit"])
    _require(report["calibration_parent_models"] == config["required_calibration_parents"] == 12 and
             report["checkpoint_sha256"] == winner["checkpoint_sha256"] and
             np.isfinite(report["conformal_scale"]) and report["conformal_scale"] > 0,
             "selected calibration must retain all 12 dedicated source parents and a positive finite frozen scale")
    validate_selection_freeze(freeze_dir)
    verify_prior_stage_binding(frozen["provenance"], frozen["runtime_contract"], repository_root)
    models = []
    for model in frozen["models"]:
        relative = f"calibration/{model['label']}.json"
        source_path = output / "selected_calibration/summary.json" if model["label"] == "development_winner" else _inside(freeze_dir, model["calibration_path"])
        _write_once(output, relative, source_path.read_bytes())
        models.append({"label": model["label"], "checkpoint_sha256": model["checkpoint_sha256"],
                       "calibration_path": relative, "calibration_sha256": sha256_file(output / relative)})
    _require(source_digest(repository_root) == source["source_sha256"], "calibration source changed during execution")
    result = {"schema_version": 1, "stage": "C-calibration-frozen-before-fresh-confirmation", "created_utc": _utc(),
        "selection_freeze_sha256": sha256_file(Path(freeze_dir) / "freeze.json"),
        "plan_sha256": frozen["plan_sha256"], "models": models,
        "source_unchanged_during_run": True, "provenance": source,
        "runtime_contract": _runtime_contract(source), "prior_stage_source_runtime_validation": prior_binding,
        "scope": "New selected-checkpoint calibration on the existing 12 calibration parents; historical 8 ID/1 OOD parents are reused development evidence, not pristine confirmation."}
    _write_once(output, "calibration_freeze.json", _json_bytes(result))
    return result


def run_selected_calibration(*, freeze_dir, repository_root, dataset_dir, output_dir):
    """Stage C with a failure receipt; failed attempts never yield a gate."""
    output = Path(output_dir)
    new_output = not output.exists()
    try:
        return _run_selected_calibration(freeze_dir=freeze_dir, repository_root=repository_root,
            dataset_dir=dataset_dir, output_dir=output_dir)
    except Exception as error:
        if new_output and output.is_dir():
            _write_once(output, "failure.json", _json_bytes({"status": "failed", "created_utc": _utc(),
                "error_type": type(error).__name__, "error": str(error),
                "fresh_confirmation_authorized": False}))
        raise


def validate_calibration_freeze(selection_dir, calibration_dir):
    frozen = validate_selection_freeze(selection_dir)
    calibration = _json(Path(calibration_dir) / "calibration_freeze.json")
    _require(calibration["stage"] == "C-calibration-frozen-before-fresh-confirmation" and
             calibration["selection_freeze_sha256"] == sha256_file(Path(selection_dir) / "freeze.json") and
             calibration["plan_sha256"] == frozen["plan_sha256"] and
             {m["label"] for m in calibration["models"]} == {m["label"] for m in frozen["models"]} and
             len(calibration["models"]) == 3, "both immutable freeze stages must precede fresh evaluation")
    _require(calibration["source_unchanged_during_run"] is True,
             "selected calibration source changed during execution")
    plan = _json(Path(selection_dir) / "plan.json")
    for entry in calibration["models"]:
        weight = next(m for m in frozen["models"] if m["label"] == entry["label"])
        path = _inside(calibration_dir, entry["calibration_path"])
        _require(entry["checkpoint_sha256"] == weight["checkpoint_sha256"] and
                 sha256_file(path) == entry["calibration_sha256"], "calibration/weight frozen hashes disagree")
        report = _json(path)
        _require(report["checkpoint_sha256"] == weight["checkpoint_sha256"] and
                 np.isfinite(report["conformal_scale"]) and report["conformal_scale"] > 0,
                 "calibration scale is invalid or tied to another checkpoint")
        if entry["label"] != "development_winner":
            _require(entry["calibration_sha256"] == weight["calibration_sha256"],
                     "legacy calibration snapshots must not be recalibrated or changed")
        else:
            _require(report["calibration_parent_models"] == 12 and report["alpha"] == plan["selected_calibration"]["alpha"] and
                     report["dataset_provenance"]["dataset_manifest_sha256"] == plan["selected_calibration"]["dataset_manifest_sha256"],
                     "new selected calibration must use the planned dedicated 12-parent role")
    return frozen, calibration


def validate_confirmation_payloads(plan, data_dir, *, published_receipt=None):
    """Validate native arrays, independent of regenerated receipt timestamps."""
    data_dir = Path(data_dir)
    local_path = data_dir / "preparation_manifest.json"
    local = _json(local_path) if local_path.is_file() else None
    if local is not None:
        _require(local["status"] == "complete" and
                 local["protocol_sha256"] == plan["old_external"]["protocol_sha256"],
                 "local preparation receipt has the wrong structural identity")
    families, files, audits = [], {}, {}
    for family in plan["families"]:
        path = _inside(data_dir, family["raw_npy_relative_path"])
        actual_sha = sha256_file(path)
        _require(actual_sha == family["raw_npy_sha256"], "native velocity payload SHA256 changed")
        crc = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                crc = zlib.crc32(chunk, crc)
        _require(f"{crc & 0xffffffff:08x}" == family["expected_crc32"], "native NPY member CRC32 changed")
        if local is not None:
            records = [f for f in local["families"] if f["name"] == family["name"]]
            _require(len(records) == 1, "local receipt omits or duplicates a frozen family")
            entry = records[0]
            _require(entry["member"] == family["member"] and entry["family_code"] == family["family_code"] and
                     entry["sha256"] == actual_sha and entry["selected_indices"] == family["excluded_old_indices"] and
                     entry["zip_member"]["crc32"] == family["expected_crc32"],
                     "local receipt changes native payload or old exclusion identities")
        batch = np.load(path, mmap_mode="r", allow_pickle=False)
        _require(list(batch.shape) == family["expected_shape"] and str(batch.dtype) == family["expected_dtype"],
                 "native velocity array shape or dtype changed")
        if published_receipt is not None:
            previous = next(f for f in published_receipt["families"] if f["name"] == family["name"])
            for old_index, expected_hash, model_id in zip(previous["selected_indices"], previous["selected_model_sha256"], previous["selected_model_ids"], strict=True):
                _require(_sha(np.ascontiguousarray(batch[old_index]).tobytes()) == expected_hash and
                         model_id == f"{family['name']}:{family['member']}:{old_index}",
                         "published native model hash or original model ID changed")
        hashes = []
        for index in family["eligible_complement_indices"]:
            raw = np.asarray(batch[index])
            _require(np.all(np.isfinite(raw)) and np.all(raw > 0), "eligible source velocities must be finite and positive")
            hashes.append(_sha(np.ascontiguousarray(raw).tobytes()))
        lookup = dict(zip(family["eligible_complement_indices"], hashes, strict=True))
        audits[family["name"]] = {
            "analysis_population_size": 400, "sample_size": 100,
            "population_unique_raw_hashes": len(set(hashes)),
            "selected_unique_raw_hashes": len({lookup[i] for i in family["confirmation_selected_indices"]}),
            "selected_model_sha256": {str(i): lookup[i] for i in family["confirmation_selected_indices"]},
            "published_old_selected_native_hashes_verified": published_receipt is not None,
            "sampling_scope": "Original index slots; repeated values remain fixed population units for design-based hypergeometric inference.",
        }
        files[str(path.resolve())] = actual_sha
        families.append((family, batch, lookup))
    if local is not None:
        files[str(local_path.resolve())] = sha256_file(local_path)
    return families, files, audits


def aggregate_gradient_metrics(exact_gradient, neural_gradient, *, threshold=1e-14):
    """Metrics of summed model gradients, never the mean of block cosines."""
    exact, neural = np.asarray(exact_gradient, dtype=float), np.asarray(neural_gradient, dtype=float)
    _require(exact.ndim == neural.ndim == 1 and exact.shape == neural.shape and
             np.all(np.isfinite(exact)) and np.all(np.isfinite(neural)), "aggregate gradients must be finite aligned vectors")
    exact_norm, neural_norm = float(np.linalg.norm(exact)), float(np.linalg.norm(neural))
    _require(np.isfinite(exact_norm) and np.isfinite(neural_norm), "aggregate gradient norm overflow")
    cosine = float(np.clip(np.dot(exact, neural) / exact_norm / neural_norm, -1, 1)) if min(exact_norm, neural_norm) > threshold else 0.0
    derivative = float(-np.dot(exact, neural) / neural_norm) if neural_norm > threshold else 0.0
    _require(np.isfinite(cosine) and np.isfinite(derivative), "aggregate gradient metric overflow")
    return {"aggregate_gradient_cosine": cosine,
            "aggregate_exact_gradient_norm": exact_norm, "aggregate_neural_gradient_norm": neural_norm,
            "aggregate_exact_gradient_zero": exact_norm <= threshold,
            "aggregate_neural_gradient_zero": neural_norm <= threshold,
            "aggregate_neural_direction_exact_derivative": derivative,
            "aggregate_neural_true_descent": bool(derivative < 0)}


def summarize_confirmation(blocks, models, plan, raw_audits):
    """Explicit N=400 complement analysis, separate from the old N=500 tool."""
    summary = {}
    analysis = plan["analysis"]
    expected_labels = {"legacy_forward", "legacy_jvp", "development_winner"}
    families = {f["name"] for f in plan["families"]}
    _require(len(blocks) == 2400 and len(models) == 600 and
             {r["family"] for r in blocks + models} == families and
             {r["checkpoint"] for r in blocks + models} == expected_labels,
             "confirmation must retain exactly both families, all three model labels, and every planned case")
    for family in plan["families"]:
        name = family["name"]
        summary[name] = {}
        expected_indices = set(family["confirmation_selected_indices"])
        for label in sorted(expected_labels):
            selected = [r for r in models if r["family"] == name and r["checkpoint"] == label]
            rows = [r for r in blocks if r["family"] == name and r["checkpoint"] == label]
            _require(len(selected) == 100 and {r["original_index"] for r in selected} == expected_indices and len(rows) == 400,
                     "every model label must retain all 100 models and 400 correlated blocks per family")
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["original_index"]].append(row)
            _require(set(grouped) == expected_indices and all(len(g) == 4 and len({(r["source_index"], r["frequency_hz"]) for r in g}) == 4 for g in grouped.values()),
                     "all four distinct source/frequency blocks are required per original model")
            events = {
                "aggregate_neural_non_descent": [not r["aggregate_neural_true_descent"] for r in selected],
                "model_any_source_calibrated_false_acceptance": [any(b["source_calibrated_false_acceptance"] for b in grouped[r["original_index"]]) for r in selected],
                "model_source_calibrated_coverage_failure": [not all(b["source_calibrated_covers"] for b in grouped[r["original_index"]]) for r in selected],
                "model_any_deterministic_false_acceptance": [any(b["deterministic_false_acceptance"] for b in grouped[r["original_index"]]) for r in selected],
                "model_deterministic_coverage_failure": [not all(b["deterministic_covers"] for b in grouped[r["original_index"]]) for r in selected],
            }
            finite_bounds = {event: finite_population_failure_summary(np.asarray(values),
                population_size=400, confidence=analysis["confidence"]) for event, values in events.items()}
            cosines = np.asarray([r["aggregate_gradient_cosine"] for r in selected])
            bootstrap = {"mean": float(cosines.mean()), "interval": None,
                         "assumption": "Conditional independent generated draws, not verified real-site inference.",
                         "status": "withheld: raw duplicate clusters"}
            if raw_audits[name]["population_unique_raw_hashes"] == 400:
                rng = np.random.default_rng(analysis["bootstrap_seed"])
                means = rng.choice(cosines, size=(analysis["bootstrap_resamples"], len(cosines)), replace=True).mean(axis=1)
                tail = (1 - analysis["confidence"]) / 2
                bootstrap.update(interval=np.quantile(means, [tail, 1 - tail]).tolist(), status="conditional generated-draw bootstrap",
                                 seed=analysis["bootstrap_seed"], n_resamples=analysis["bootstrap_resamples"])
            summary[name][label] = {
                "analysis_population_size": 400, "sampled_original_index_slots": 100, "correlated_blocks": 400,
                "primary_mean_actual_aggregate_gradient_cosine": float(cosines.mean()),
                "primary_actual_aggregate_neural_true_descent_fraction": float(np.mean([r["aggregate_neural_true_descent"] for r in selected])),
                "conditional_model_bootstrap_aggregate_cosine": bootstrap,
                "fixed_complement_hypergeometric_bounds": finite_bounds,
                "block_gradient_cosine_median_secondary": float(np.median([r["gradient_cosine"] for r in rows])),
                "source_calibrated_accepted_blocks": sum(r["source_calibrated_accepts"] for r in rows),
                "source_calibrated_false_accepted_blocks": sum(r["source_calibrated_false_acceptance"] for r in rows),
                "deterministic_false_accepted_blocks": sum(r["deterministic_false_acceptance"] for r in rows),
            }
    return summary


def _write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_gradient_confirmation(*, plan_path, repository_root, selection_dir, calibration_dir, data_dir, output_dir):
    """Stage D: all three frozen models on all fresh complement-index cases."""
    import torch

    root = Path(repository_root)
    plan = validate_confirmation_plan(plan_path, root)
    frozen, calibration = validate_calibration_freeze(selection_dir, calibration_dir)
    prior_bindings = {
        "selection_freeze": verify_prior_stage_binding(frozen["provenance"], frozen["runtime_contract"], root),
        "calibration_freeze": verify_prior_stage_binding(calibration["provenance"], calibration["runtime_contract"], root),
    }
    _require(frozen["plan_sha256"] == sha256_file(plan_path), "evaluation plan differs from both frozen stages")
    published = _json(_inside(root, plan["old_external"]["published_preparation_manifest_path"]))
    families, data_hashes, audits = validate_confirmation_payloads(plan, data_dir, published_receipt=published)
    # All selection/calibration/raw input checks precede model construction or prediction.
    output = _new_directory(output_dir)
    torch.set_num_threads(1)
    provenance = environment_record(root)
    bindings = {str(Path(plan_path).resolve()): sha256_file(plan_path),
                str((Path(selection_dir) / "freeze.json").resolve()): sha256_file(Path(selection_dir) / "freeze.json"),
                str((Path(calibration_dir) / "calibration_freeze.json").resolve()): sha256_file(Path(calibration_dir) / "calibration_freeze.json")}
    paths, scales = {}, {}
    for model in frozen["models"]:
        label = model["label"]
        path = _inside(selection_dir, model["checkpoint_path"])
        paths[label] = path
        bindings[str(path.resolve())] = model["checkpoint_sha256"]
        entry = next(c for c in calibration["models"] if c["label"] == label)
        calibration_path = _inside(calibration_dir, entry["calibration_path"])
        bindings[str(calibration_path.resolve())] = entry["calibration_sha256"]
        scales[label] = float(_json(calibration_path)["conformal_scale"])
    bindings.update(data_hashes)
    status_path = output / "run_status.json"
    status = {"status": "running", "started_utc": _utc(), "completed_blocks": 0,
              "input_sha256": bindings, "provenance": provenance}
    status_path.write_bytes(_json_bytes(status))
    blocks, models, exact_vectors, neural_vectors = [], [], [], []
    current = None
    try:
        surrogates = {label: TrainedFNOWavefieldSurrogate.load(path, device="cpu") for label, path in paths.items()}
        _require(all(s.shape == (24, 24) for s in surrogates.values()), "frozen models must share the planned 24x24 grid")
        geometry = SurveyGeometry.from_grid((24, 24), n_sources=6, n_receivers=20, source_depth=2, receiver_depth=2, margin=2)
        sources = (geometry.source_indices[1], geometry.source_indices[-2])
        truth, physics = (Helmholtz2D(shape=(24, 24), spacing=.05, damping_width=3, damping_strength=2) for _ in range(2))
        for family, batch, raw_hashes in families:
            for index in family["confirmation_selected_indices"]:
                current = {"family": family["name"], "original_index": index}
                m_true, m_eval = prepare_external_models(batch[index])
                truth.clear_solver_cache()
                physics.clear_solver_cache()
                reference, constants, survey = {}, {}, []
                for frequency in (3.0, 4.0):
                    for source in sources:
                        q = truth.source(source)
                        observed = truth.restrict(truth.solve_state(m_true, q, frequency), geometry.receiver_indices)
                        value, gradient, state, _ = physics.objective_and_gradient(m_eval, q, observed, geometry.receiver_indices, frequency)
                        reference[source, frequency] = (q, observed, value, gradient, state)
                        survey.append(SurveyBlock(int(source), frequency, observed, f"{source}@{frequency}"))
                    matrix = physics.receiver_resolvent_matrix(m_eval, frequency, geometry.receiver_indices)
                    constants[frequency] = (matrix, physics.receiver_resolvent_norm(m_eval, frequency, geometry.receiver_indices, receiver_resolvent_matrix=matrix))
                exact_total = np.sum([entry[3] for entry in reference.values()], axis=0)
                exact_objective = float(sum(entry[2] for entry in reference.values()))
                for label, surrogate in surrogates.items():
                    current = {**current, "checkpoint": label}
                    neural_block_gradients, neural_objectives, case_rows = [], [], []
                    for block in survey:
                        q, observed, value, exact_gradient, exact_state = reference[block.source_index, block.frequency_hz]
                        neural_value, neural_gradient, neural_state, _ = surrogate.objective_and_gradient(m_eval, q, observed, geometry.receiver_indices, block.frequency_hz)
                        norm = float(np.linalg.norm(neural_gradient))
                        _require(np.isfinite(norm) and np.all(np.isfinite(neural_gradient)) and np.all(np.isfinite(neural_state)), "nonfinite neural output")
                        direction = -neural_gradient / norm if norm > 1e-14 else np.zeros_like(neural_gradient)
                        matrix, alpha = constants[block.frequency_hz]
                        kappa = physics.directional_receiver_tangent_resolvent_norm(m_eval, block.frequency_hz, geometry.receiver_indices, direction, receiver_resolvent_matrix=matrix)
                        certificate = certify_direction(physics=physics, surrogate=surrogate, m=m_eval, q=q, observed=observed,
                            receiver_indices=geometry.receiver_indices, frequency_hz=block.frequency_hz, direction=direction,
                            stability_mode="directional", receiver_resolvent_matrix=matrix, receiver_resolvent_norm=alpha,
                            directional_tangent_resolvent_norm=kappa)
                        derivative = float(np.dot(exact_gradient, direction))
                        realized = abs(derivative - certificate.surrogate_directional_derivative)
                        bound, scaled = certificate.directional_error_bound, scales[label] * certificate.directional_error_bound
                        accepts = certificate.surrogate_directional_derivative + scaled < 0
                        deterministic_accepts = certificate.certified_descent
                        row = {"family": family["name"], "checkpoint": label, "original_index": index,
                            "model_id": f"{family['name']}:{family['member']}:{index}", "raw_model_sha256": raw_hashes[index],
                            "source_index": int(block.source_index), "frequency_hz": block.frequency_hz,
                            "exact_objective": value, "neural_objective": neural_value,
                            "forward_receiver_relative_l2": relative_l2(physics.restrict(exact_state, geometry.receiver_indices), physics.restrict(neural_state, geometry.receiver_indices)),
                            "gradient_relative_l2": relative_l2(exact_gradient, neural_gradient), "gradient_cosine": cosine_similarity(exact_gradient, neural_gradient),
                            "exact_directional_derivative": derivative, "surrogate_directional_derivative": certificate.surrogate_directional_derivative,
                            "deterministic_bound": bound, "source_calibrated_bound": scaled, "source_calibration_scale": scales[label],
                            "realized_directional_error": realized, "deterministic_covers": bool(realized <= bound + 1e-12),
                            "source_calibrated_covers": bool(realized <= scaled + 1e-12), "deterministic_accepts": bool(deterministic_accepts),
                            "source_calibrated_accepts": bool(accepts), "deterministic_false_acceptance": bool(deterministic_accepts and derivative >= 0),
                            "source_calibrated_false_acceptance": bool(accepts and derivative >= 0)}
                        _require(all(np.isfinite(v) for v in row.values() if isinstance(v, (float, np.floating))), "nonfinite block metric")
                        case_rows.append(row)
                        neural_block_gradients.append(neural_gradient)
                        neural_objectives.append(neural_value)
                    # Shared-input autograd of the actual summed four-block objective.
                    aggregate_value, aggregate_gradient, _, _ = surrogate.survey_objective_and_gradient(m_eval, survey, geometry.receiver_indices)
                    summed = np.sum(neural_block_gradients, axis=0)
                    agreement = float(np.linalg.norm(summed - aggregate_gradient) / max(np.linalg.norm(summed), 1e-15))
                    _require(agreement <= 2e-4 and math.isclose(aggregate_value, sum(neural_objectives), rel_tol=5e-6, abs_tol=1e-10),
                             "shared-input aggregate and summed block gradients disagree")
                    models.append({"family": family["name"], "checkpoint": label, "original_index": index,
                        "model_id": case_rows[0]["model_id"], "raw_model_sha256": raw_hashes[index], "n_blocks": 4,
                        "aggregate_exact_objective": exact_objective, "aggregate_neural_objective": aggregate_value,
                        "aggregate_vs_block_sum_relative_gradient_error": agreement,
                        "secondary_mean_block_gradient_cosine": float(np.mean([r["gradient_cosine"] for r in case_rows])),
                        **aggregate_gradient_metrics(exact_total, aggregate_gradient)})
                    exact_vectors.append(exact_total)
                    neural_vectors.append(aggregate_gradient)
                    blocks.extend(case_rows)
                _write_rows(output / "blocks.csv", blocks)
                _write_rows(output / "models.csv", models)
                status.update(completed_blocks=len(blocks), last_completed_model=current)
                status_path.write_bytes(_json_bytes(status))
        grouped = summarize_confirmation(blocks, models, plan, audits)
        verify_prior_stage_binding(frozen["provenance"], frozen["runtime_contract"], root)
        verify_prior_stage_binding(calibration["provenance"], calibration["runtime_contract"], root)
        source_same = source_digest(root) == provenance["source_sha256"]
        inputs_same = all(sha256_file(path) == checksum for path, checksum in bindings.items())
        _require(source_same and inputs_same, "source or frozen input changed during confirmation")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, exact_gradients=np.asarray(exact_vectors), neural_gradients=np.asarray(neural_vectors),
            family=np.asarray([row["family"] for row in models]), checkpoint=np.asarray([row["checkpoint"] for row in models]),
            original_index=np.asarray([row["original_index"] for row in models]), model_id=np.asarray([row["model_id"] for row in models]))
        _write_once(output, "aggregate_gradients.npz", buffer.getvalue())
        summary = {"schema_version": 1, "status": "complete", "completed_utc": _utc(),
            "n_blocks": len(blocks), "n_model_checkpoint_evaluations": len(models),
            "families": grouped, "raw_sampling_audits": audits, "provenance": provenance,
            "source_unchanged_during_run": True, "input_files_unchanged_during_run": True,
            "input_sha256": bindings, "aggregate_vectors_sha256": sha256_file(output / "aggregate_gradients.npz"),
            "prior_stage_source_runtime_validation": prior_bindings, "runtime_contract": _runtime_contract(provenance),
            "offline_physics_costs": {"truth": truth.cost_snapshot(), "reference_and_certificates": physics.cost_snapshot()},
            "scope": "Prospectively chosen fresh original-index outcomes from two known generated batches, with N400 complement bounds. All three fixed models reported; no held-out model selection, real-site/field-data, trajectory, or acceleration claim.",
            "source_calibrated_gate_scope": "Descriptive external/OOD transfer of frozen source-domain scales; no nominal conformal coverage guarantee.",
            "historical_source_test_scope": "Selected calibration's historical 8 ID/1 OOD test-parent report is development reuse, not this fresh-index confirmation."}
        _write_once(output, "summary.json", _json_bytes(summary))
        status.update(status="complete", completed_blocks=len(blocks), source_unchanged_during_run=True, input_files_unchanged_during_run=True)
        status_path.write_bytes(_json_bytes(status))
        return summary
    except Exception as error:
        status.update(status="failed", completed_blocks=len(blocks), failed_unit=current,
                      error_type=type(error).__name__, error=str(error))
        status_path.write_bytes(_json_bytes(status))
        raise
