"""Training-only POD bases and development-only common validation controls."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wavecert.experiments.gradient_development import (
    _array_digest,
    _atomic_bytes,
    _write_json,
    parent_weighted_mean,
)
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.pod import PODGalerkinSurrogate


def validate_pod_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != 1 or protocol["ranks"] != [8, 16, 32, 64]:
        raise ValueError("POD controls require the frozen four ranks")
    families = protocol["basis_families"]
    if [(row["name"], row["include_training_tangents"]) for row in families] != [("state", False), ("state+tangent", True)]:
        raise ValueError("both frozen POD basis families must be retained in order")
    for row in families:
        if not row["artifact_label"].replace("_", "").isalnum():
            raise ValueError("POD basis artifact labels must be safe path components")
    if protocol["basis_construction"]["maximum_saved_rank"] != max(protocol["ranks"]):
        raise ValueError("maximum saved basis rank must match the frozen prefix ranks")
    floor = protocol["basis_construction"]["squared_norm_floor"]
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("POD snapshot energy floor must be finite and positive")


def load_teacher_inputs(protocol: dict, teacher_study: Path) -> tuple[dict, dict, dict]:
    """Read only pinned train/validation-derived archives, never held-out data."""
    expected = protocol["teacher_inputs"]
    fixed_files = {
        "protocol.json": expected["protocol_sha256"],
        "teacher_labels.json": expected["teacher_labels_sha256"],
        "input_audit.json": expected["input_audit_sha256"],
    }
    verified_files = {}
    for name, checksum in fixed_files.items():
        path = teacher_study / name
        if sha256_file(path) != checksum:
            raise ValueError(f"pinned teacher study input changed: {name}")
        verified_files[name] = checksum
    teacher_protocol = json.loads((teacher_study / "protocol.json").read_text())
    if protocol["basis_construction"]["squared_norm_floor"] != teacher_protocol["loss"]["squared_norm_denominator_floor"] or protocol["validation"]["gradient_zero_threshold"] != teacher_protocol["validation_inverse_cases"]["gradient_norm_zero_threshold"]:
        raise ValueError("POD controls must retain the teacher comparison's physical floors and gradient convention")
    teacher_records = json.loads((teacher_study / "teacher_labels.json").read_text())
    audit = json.loads((teacher_study / "input_audit.json").read_text())
    if audit["opened_array_roles"] != ["train", "validation"] or audit["calibration_or_test_arrays_loaded"] or audit["external_test_arrays_loaded"]:
        raise ValueError("teacher study must be train/validation-only")
    if not audit["parent_and_slice_audit"]["parent_isolation_passed"]:
        raise ValueError("teacher study parent isolation did not pass")
    arrays = {}
    for role in ("train", "validation", "validation_inverse_cases"):
        item = expected["archives"][role]
        path = teacher_study / item["path"]
        if sha256_file(path) != item["sha256"] or item["sha256"] != teacher_records[role]["artifact_sha256"]:
            raise ValueError(f"pinned {role} teacher label archive changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays[role] = {name: archive[name].copy() for name in archive.files}
        verified_files[item["path"]] = item["sha256"]
        for name, value in arrays[role].items():
            if _array_digest(value) != teacher_records[role]["array_sha256"][name]:
                raise ValueError(f"{role} teacher array identity changed: {name}")
    identities = {}
    for role, required_count in (("train", expected["train_parents"]), ("validation", expected["validation_parents"])):
        labels = arrays[role]
        parents, slices = set(map(str, labels["parent_ids"])), set(map(str, labels["slice_ids"]))
        recorded = audit["parent_and_slice_audit"]["roles"][role]
        if len(parents) != required_count or parents != set(recorded["parent_models"]):
            raise ValueError(f"{role} parent identities differ from the pinned audit")
        if len(slices) != recorded["unique_slices"] or set(labels["state_types"]) != {"raw", "smoothed"}:
            raise ValueError(f"{role} slice/state identities differ from pinned development design")
        identities[role] = {"parent_ids": sorted(parents), "slice_ids": sorted(slices),
                            "state_examples": len(labels["models"])}
    if set(identities["train"]["parent_ids"]) & set(identities["validation"]["parent_ids"]) or set(identities["train"]["slice_ids"]) & set(identities["validation"]["slice_ids"]):
        raise ValueError("training and validation parents/slices overlap")
    inverse = arrays["validation_inverse_cases"]
    if set(inverse["parent_ids"]) != set(identities["validation"]["parent_ids"]) or set(inverse["slice_ids"]) != set(identities["validation"]["slice_ids"]):
        raise ValueError("inverse cases must contain every pinned validation slice")
    return arrays, teacher_protocol, {
        "teacher_study": str(teacher_study.resolve()), "verified_file_sha256": verified_files,
        "identities": identities, "shared_offline_teacher_label_generation": teacher_records,
        "cost_scope": "Teacher state/tangent/inverse-gradient labels are reused shared offline work, not free computation",
        "opened_archive_roles": ["train", "validation", "validation_inverse_cases"],
        "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False,
    }


def normalized_real_snapshots(fields: np.ndarray, tangents: np.ndarray | None, floor: float) -> np.ndarray:
    """Normalize a complex snapshot jointly, then concatenate its real/imag parts."""
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("snapshot energy floor must be finite and positive")
    matrices = []
    for snapshots in (fields,) if tangents is None else (fields, tangents):
        values = np.asarray(snapshots, dtype=np.float64)
        if values.ndim < 3 or not len(values) or values.shape[-1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError("POD snapshots must have finite physical real/imaginary channels")
        flat = values.reshape(len(values), -1, 2)
        energy = np.sum(flat**2, axis=(1, 2))
        if not np.all(np.isfinite(energy)):
            raise FloatingPointError("POD snapshot energy overflowed")
        scale = np.sqrt(np.maximum(energy, floor))
        normalized = flat / scale[:, None, None]
        matrices.extend([normalized[..., 0].T, normalized[..., 1].T])
    if tangents is not None and fields.shape != tangents.shape:
        raise ValueError("state and tangent training snapshot shapes must match")
    return np.concatenate(matrices, axis=1)


def fit_normalized_real_pod(fields: np.ndarray, tangents: np.ndarray | None, *, max_rank: int, floor: float):
    matrix = normalized_real_snapshots(fields, tangents, floor)
    if not 1 <= max_rank <= min(matrix.shape):
        raise ValueError("requested POD rank exceeds snapshot matrix dimensions")
    basis, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    basis = np.ascontiguousarray(basis[:, :max_rank])
    return basis, singular_values, {
        "svd_evaluations": 1, "snapshot_matrix_shape": list(matrix.shape),
        "snapshot_matrix_sha256": _array_digest(matrix), "basis_sha256": _array_digest(basis),
        "snapshot_normalization": "joint complex-L2 snapshot normalization, not separate channel normalization",
        "state_and_tangent_snapshots_used": tangents is not None,
        "numerical_rank_default_tolerance": int(np.sum(singular_values > singular_values[0] * max(matrix.shape) * np.finfo(float).eps)),
        "singular_values": singular_values.tolist(),
    }


def _relative_squared(prediction, target, floor: float) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    return float(np.vdot(error, error).real / max(float(np.vdot(target, target).real), floor))


def evaluate_surrogate_validation(surrogate, labels: dict, cases: dict, teacher_protocol: dict) -> dict:
    """Same physical three-component parent-weighted criterion as FNO validation."""
    physics = Helmholtz2D(
        shape=tuple(teacher_protocol["teacher"]["shape"]), spacing=teacher_protocol["teacher"]["spacing"],
        damping_width=teacher_protocol["teacher"]["damping_width"], damping_strength=teacher_protocol["teacher"]["damping_strength"],
    )
    receivers = np.asarray(labels["receiver_indices"], dtype=int)
    floor = teacher_protocol["loss"]["squared_norm_denominator_floor"]
    losses = defaultdict(list)
    for index, model in enumerate(labels["models"]):
        m = model.astype(float).reshape(-1)
        direction = labels["directions"][index].astype(float).reshape(-1)
        q = physics.source(int(labels["source_indices"][index]))
        frequency = float(labels["frequencies_hz"][index])
        state, tangent = surrogate.state_and_jvp(m, direction, q, frequency)
        truth = labels["fields"][index]
        truth = (truth[..., 0] + 1j * truth[..., 1]).reshape(-1).astype(np.complex128)
        true_tangent = labels["tangents"][index]
        true_tangent = (true_tangent[..., 0] + 1j * true_tangent[..., 1]).reshape(-1).astype(np.complex128)
        losses["full_field_relative_squared_l2"].append(_relative_squared(state, truth, floor))
        losses["receiver_relative_squared_l2"].append(_relative_squared(state[receivers], truth[receivers], floor))
        losses["jvp_relative_squared_l2"].append(_relative_squared(tangent, true_tangent, floor))
    metrics = {key: parent_weighted_mean(np.asarray(values), labels["parent_ids"]) for key, values in losses.items()}
    gradients = np.zeros_like(cases["exact_gradients"], dtype=np.float64)
    for index, case in enumerate(cases["block_case_indices"]):
        observations = cases["observations"][index]
        observations = observations[:, 0] + 1j * observations[:, 1]
        _, gradient, _, _ = surrogate.objective_and_gradient(
            cases["models"][case].astype(float).reshape(-1), physics.source(int(cases["source_indices"][index])),
            observations, tuple(cases["receiver_indices"].tolist()), float(cases["frequencies_hz"][index]),
        )
        gradients[case] += gradient
    if not np.all(np.isfinite(gradients)):
        raise FloatingPointError("nonfinite reduced-model validation gradient")
    threshold = teacher_protocol["validation_inverse_cases"]["gradient_norm_zero_threshold"]
    inverse_rows = []
    for index, (exact, reduced) in enumerate(zip(cases["exact_gradients"], gradients, strict=True)):
        exact_norm, reduced_norm = float(np.linalg.norm(exact)), float(np.linalg.norm(reduced))
        if not np.isfinite(exact_norm) or not np.isfinite(reduced_norm):
            raise FloatingPointError("validation gradient norm overflowed")
        cosine = float(np.clip(np.dot(exact, reduced) / (exact_norm * reduced_norm), -1, 1)) if min(exact_norm, reduced_norm) > threshold else 0.0
        derivative = float(-np.dot(exact, reduced) / reduced_norm) if reduced_norm > threshold else 0.0
        inverse_rows.append({"slice_id": str(cases["slice_ids"][index]), "model_id": str(cases["parent_ids"][index]),
                             "gradient_cosine": cosine, "exact_gradient_norm": exact_norm, "neural_gradient_norm": reduced_norm,
                             "exact_gradient_zero": exact_norm <= threshold, "neural_gradient_zero": reduced_norm <= threshold,
                             "neural_direction_exact_derivative": derivative, "neural_true_descent": derivative < 0})
    # Preserve the FNO helper's key names for direct criterion comparison;
    # here "neural" keys refer to the evaluated surrogate, which is a ROM.
    for key in ("gradient_cosine", "exact_gradient_norm", "neural_gradient_norm", "neural_true_descent"):
        metrics[f"aggregate_{key}"] = parent_weighted_mean(np.asarray([row[key] for row in inverse_rows], dtype=float), cases["parent_ids"])
    metrics["direction_loss"] = 1 - metrics["aggregate_gradient_cosine"]
    metrics["selection_score"] = metrics["receiver_relative_squared_l2"] + metrics["jvp_relative_squared_l2"] + metrics["direction_loss"]
    metrics["inverse_cases"] = inverse_rows
    metrics["parent_components"] = {
        str(parent): {key: float(np.mean(np.asarray(value)[labels["parent_ids"] == parent])) for key, value in losses.items()}
        for parent in sorted(set(labels["parent_ids"]))
    }
    for parent, values in metrics["parent_components"].items():
        rows = [row for row in inverse_rows if row["model_id"] == parent]
        values.update({f"aggregate_{key}": float(np.mean([row[key] for row in rows]))
                       for key in ("gradient_cosine", "neural_true_descent", "exact_gradient_norm", "neural_gradient_norm")})
    metrics["state_type_components"] = {
        str(state): {key: parent_weighted_mean(np.asarray(value)[labels["state_types"] == state], labels["parent_ids"][labels["state_types"] == state])
                     for key, value in losses.items()} for state in ("raw", "smoothed")
    }
    return metrics


def run_pod_development(protocol_path: Path, teacher_study: Path, output: Path, *, root: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[3] if root is None else root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root) or output == root or output.is_relative_to(teacher_study.resolve()):
        raise ValueError("POD outputs must stay inside this isolated repository and outside the read-only teacher study")
    if output.exists() and any(output.iterdir()):
        raise ValueError("POD output must be new or empty; preserve existing evidence")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_pod_protocol(protocol)
    arrays, teacher_protocol, inputs = load_teacher_inputs(protocol, teacher_study)
    provenance = environment_record(root)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output / "protocol.json", protocol_bytes)
    _write_json(output / "input_provenance.json", inputs)
    _write_json(output / "status.json", {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat()})
    results, families = [], {}
    try:
        for family in protocol["basis_families"]:
            training = arrays["train"]
            basis, singular_values, fit_record = fit_normalized_real_pod(
                training["fields"], training["tangents"] if family["include_training_tangents"] else None,
                max_rank=max(protocol["ranks"]), floor=protocol["basis_construction"]["squared_norm_floor"],
            )
            teacher = teacher_protocol["teacher"]
            physics = Helmholtz2D(shape=tuple(teacher["shape"]), spacing=teacher["spacing"],
                                  damping_width=teacher["damping_width"], damping_strength=teacher["damping_strength"])
            path = output / "bases" / f"{family['artifact_label']}_rank64.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = PODGalerkinSurrogate(physics, basis)
            saved.save(path, metadata={"family": family["name"], "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                                       "training_archive_sha256": protocol["teacher_inputs"]["archives"]["train"]["sha256"],
                                       "fit": fit_record})
            fit_record.update({"artifact": str(path.relative_to(output)), "artifact_sha256": sha256_file(path),
                               "saved_rank": basis.shape[1], "singular_value_count": len(singular_values),
                               "saved_basis_operator_projection_cost": saved.offline_projection_cost})
            families[family["name"]] = fit_record
            for rank in protocol["ranks"]:
                surrogate = PODGalerkinSurrogate(
                    physics, basis[:, :rank], factorization_cache_size=protocol["galerkin_model"]["factorization_cache_size"],
                    mass_cache_size=protocol["galerkin_model"]["mass_cache_size"],
                    orthogonality_tolerance=protocol["basis_construction"]["basis_orthonormality_tolerance"],
                )
                surrogate.reset_cost_counters(clear_cache=True)
                metrics = evaluate_surrogate_validation(surrogate, arrays["validation"], arrays["validation_inverse_cases"], teacher_protocol)
                row = {"family": family["name"], "rank": rank, "status": "complete", "validation_metrics": metrics,
                       "basis_artifact": fit_record["artifact"], "basis_artifact_sha256": fit_record["artifact_sha256"],
                       "basis_prefix_sha256": _array_digest(basis[:, :rank]), "offline_operator_projection_cost": surrogate.offline_projection_cost,
                       "validation_reduced_model_cost": surrogate.cost_snapshot()}
                results.append(row)
                _write_json(output / "candidates.json", results)
                print(json.dumps({"family": family["name"], "rank": rank, "score": metrics["selection_score"],
                                  "gradient_cosine": metrics["aggregate_gradient_cosine"]}), flush=True)
        _, _, end_inputs = load_teacher_inputs(protocol, teacher_study)
        end_source = source_digest(root)
        unchanged = provenance["source_sha256"] == end_source and protocol_path.read_bytes() == protocol_bytes
        inputs_unchanged = inputs["verified_file_sha256"] == end_inputs["verified_file_sha256"]
        if len(results) != 8 or any(row["status"] != "complete" for row in results):
            raise ValueError("all eight fixed POD candidates must complete")
        order = [row["name"] for row in protocol["basis_families"]]
        winner = min(results, key=lambda row: (row["validation_metrics"]["selection_score"], order.index(row["family"]), row["rank"]))
        selection = {"family": winner["family"], "rank": winner["rank"], "validation_selection_score": winner["validation_metrics"]["selection_score"],
                     "basis_artifact": winner["basis_artifact"], "basis_artifact_sha256": winner["basis_artifact_sha256"],
                     "basis_prefix_sha256": winner["basis_prefix_sha256"], "frozen_for_heldout_evaluation": False,
                     "changes_existing_fno_selection": False}
        summary = {"schema_version": 1, "status": "complete" if unchanged and inputs_unchanged else "stale-source-or-inputs",
                   "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(), "inputs": inputs, "basis_families": families,
                   "candidates": results, "selection": selection, "provenance": provenance, "source_sha256_at_end": end_source,
                   "source_unchanged_during_run": unchanged, "inputs_unchanged_during_run": inputs_unchanged,
                   "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False, "timing_claims": False,
                   "scope": "Classical train/validation-only POD Galerkin controls; reduced-model exact derivatives are not fine-PDE certificates, reduced RHS are not fine-grid RHS equivalents"}
        _write_json(output / "selection.json", selection)
        _write_json(output / "summary.json", summary)
        _write_json(output / "status.json", {"status": summary["status"]})
        return summary
    except BaseException as exc:
        _write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                              "completed_candidates": [{"family": row["family"], "rank": row["rank"]} for row in results]})
        raise
