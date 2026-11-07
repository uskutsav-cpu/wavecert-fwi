"""Fixed, exploratory validation reproduction of three coarse-adjoint proposals.

This runner opens only the existing compact validation fixture and frozen POD
basis. Fine solves are offline checks on a separate solver. No training,
selection, held-out evaluation, timing comparison or inversion is performed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wavecert.certificates.coarse_adjoint import METHODS, PODCoarseAdjointCertifier
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.provenance import environment_record, source_digest
from wavecert.surrogates.pod import PODGalerkinSurrogate

_PHYSICS = dict(shape=[24, 24], spacing=.05, damping_width=3, damping_strength=2.0)
_SOURCES = [50, 54, 58, 61, 65, 69]
_FREQUENCIES = [2.5, 3.0, 3.5, 4.0]
_IDENTITIES = [
    ("gom_148", "gom_148_il_0236"), ("gom_148", "gom_148_xl_0768"),
    ("gom_151", "gom_151_il_0247"), ("gom_151", "gom_151_xl_0857"),
    ("seam_005", "seam_005_il_0596"), ("seam_005", "seam_005_xl_0795"),
    ("seam_011", "seam_011_il_0522"), ("seam_011", "seam_011_xl_0866"),
]
_ARRAYS = {
    "models": ((8, 24, 24), "float32"), "parent_ids": ((8,), "<U8"),
    "slice_ids": ((8,), "<U16"), "block_case_indices": ((192,), "int64"),
    "observations": ((192, 20, 2), "float32"), "source_indices": ((192,), "int64"),
    "frequencies_hz": ((192,), "float32"), "receiver_indices": ((20,), "int64"),
    "exact_gradients": ((8, 576), "float64"), "exact_objectives": ((8,), "float64"),
}


def validate_coarse_adjoint_protocol(protocol: dict) -> None:
    """Fail closed on changes to the fixed development cohort and methods."""
    fixed = {
        "schema_version": 1, "study_role": "exploratory_validation_reproduction",
        "methods": list(METHODS), "spectral_coarse_rank": 96, "pod_rank": 64,
        "physics": _PHYSICS, "source_indices": _SOURCES,
        "frequencies_hz": _FREQUENCIES, "receiver_indices": list(range(50, 70)),
        "direction_rule": "unit negative sum of all 24 physical block gradients; zero if norm <= 1e-14",
        "aggregation_rule": "sum all 24 directional error bounds; unavailable if any block unavailable",
        "numerical_check_relative_tolerance": 1e-9,
        "numerical_check_absolute_tolerance": 1e-10,
        "timing_claims": False, "heldout_arrays_loaded": False,
    }
    for key, value in fixed.items():
        if protocol.get(key) != value:
            raise ValueError(f"fixed coarse-adjoint protocol mismatch: {key}")
    expected = [{"parent_id": p, "slice_id": s} for p, s in _IDENTITIES]
    if protocol.get("identities") != expected:
        raise ValueError("all eight original validation slices must remain in their fixed order")
    for label in ("fixture", "fixture_provenance", "basis", "pod_input_provenance", "pod_summary", "pod_source"):
        entry = protocol.get("inputs", {}).get(label, {})
        digest = entry.get("sha256", "")
        if (not isinstance(entry.get("path"), str) or Path(entry["path"]).is_absolute()
                or ".." in Path(entry["path"]).parts or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"invalid pinned input: {label}")


def load_coarse_adjoint_inputs(protocol: dict, root: Path) -> tuple[dict, np.ndarray, dict]:
    """Read, validate and bind inputs without model predictions or output writes."""
    validate_coarse_adjoint_protocol(protocol)
    hashes = {}
    payloads = {}
    for label, entry in protocol["inputs"].items():
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("pinned input escapes the repository")
        # Bind hashes and deserialization to the same read-only byte snapshot.
        # These six compact inputs total well below a megabyte.
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"input SHA-256 mismatch: {label}")
        payloads[label], hashes[label] = payload, actual
    with np.load(io.BytesIO(payloads["fixture"]), allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAYS):
            raise ValueError("validation fixture array schema changed")
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, (shape, dtype) in _ARRAYS.items():
        array = arrays[key]
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"validation fixture shape/dtype mismatch: {key}")
        if array.dtype.kind in "fci" and not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite validation fixture: {key}")
    if (np.any(arrays["models"] <= 0) or np.any(arrays["exact_objectives"] < 0)
            or list(zip(arrays["parent_ids"].tolist(), arrays["slice_ids"].tolist(), strict=True)) != _IDENTITIES
            or not np.array_equal(arrays["receiver_indices"], protocol["receiver_indices"])):
        raise ValueError("validation fixture identities, model domain or receivers changed")
    indices = arrays["block_case_indices"]
    if set(indices.tolist()) != set(range(8)):
        raise ValueError("validation block case indices must include exactly all eight cases")
    for i in range(8):
        selected = np.flatnonzero(indices == i)
        pairs = [(int(arrays["source_indices"][j]), float(arrays["frequencies_hz"][j])) for j in selected]
        if len(pairs) != 24 or set(pairs) != {(s, f) for s in _SOURCES for f in _FREQUENCIES}:
            raise ValueError("each validation slice must retain all 24 distinct source/frequency blocks")
    with np.load(io.BytesIO(payloads["basis"]), allow_pickle=False) as archive:
        basis = archive["basis"].copy()
        config = json.loads(str(archive["physics_config"]))
        basis_metadata = json.loads(str(archive["metadata"]))
    if (basis.shape != (576, 64) or basis.dtype != np.float64 or np.iscomplexobj(basis)
            or not np.all(np.isfinite(basis)) or config != _PHYSICS):
        raise ValueError("frozen POD basis/operator mismatch")
    fixture_provenance = json.loads(payloads["fixture_provenance"])
    teacher = json.loads(payloads["pod_input_provenance"])
    pod_summary = json.loads(payloads["pod_summary"])
    if (fixture_provenance["archive_sha256"] != hashes["fixture"]
            or fixture_provenance["source_provenance_sha256"] != hashes["pod_input_provenance"]
            or fixture_provenance["identities"] != protocol["identities"]
            or fixture_provenance["physics"] != _PHYSICS
            or teacher["verified_file_sha256"]["labels/validation_inverse_cases.npz"] != hashes["fixture"]
            or teacher["calibration_or_test_arrays_loaded"] or teacher["external_test_arrays_loaded"]
            or pod_summary["selection"]["basis_artifact_sha256"] != hashes["basis"]
            or pod_summary["selection"]["rank"] != 64
            or pod_summary["selection"]["family"] != "state+tangent"):
        raise ValueError("fixture/POD provenance binding failed")
    audit = {
        "verified_file_sha256": hashes, "identities": protocol["identities"],
        "opened_array_roles": ["existing_validation_inverse_cases", "frozen_training_fit_POD_basis"],
        "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False,
        "shared_offline_teacher_label_generation": teacher["shared_offline_teacher_label_generation"],
        "teacher_cost_scope": teacher["cost_scope"], "frozen_basis_fit_metadata": basis_metadata,
        "frozen_basis_selection": pod_summary["selection"], "fixture_provenance": fixture_provenance,
    }
    return arrays, basis, audit


def _write_json(path, record):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()} for row in rows)


def _physics():
    return Helmholtz2D(**{**_PHYSICS, "shape": tuple(_PHYSICS["shape"])})


def _within(error, bound, relative, absolute):
    return (error is not None and bound is not None
            and error <= bound+absolute+relative*max(error, bound))


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _ratio(numerator, denominator):
    return _finite(numerator/denominator) if denominator != 0 else None


def _cosine(first, second):
    if first is None or second is None or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return None
    # Normalize before taking the dot product, without an overflowing product
    # of norms. The data here are ordinary-scale, but abstentions remain valid.
    scale1, scale2 = float(np.max(np.abs(first))), float(np.max(np.abs(second)))
    if scale1 == 0 or scale2 == 0:
        return None
    left, right = first/scale1, second/scale2
    return _finite(np.dot(left/np.linalg.norm(left), right/np.linalg.norm(right)))


def _summarize(rows):
    summaries = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        parents = {}
        for parent in sorted({row["parent_id"] for row in group}):
            slices = [row for row in group if row["parent_id"] == parent]
            cosines = [row["gradient_cosine"] for row in slices if row["gradient_cosine"] is not None]
            parents[parent] = {"n_slices": len(slices), "certified_fraction": float(np.mean([row["certified_descent"] for row in slices])),
                "gradient_cosine_mean": float(np.mean(cosines)) if len(cosines) == len(slices) else None}
        cosine_values = [value["gradient_cosine_mean"] for value in parents.values()]
        summaries[method] = {"n_slices": len(group), "n_parents": len(parents),
            "available": sum(row["available"] for row in group),
            "certified_descent": sum(row["certified_descent"] for row in group),
            "true_descent": sum(row["exact_directional_derivative"] < 0 for row in group),
            "false_certifications": sum(row["certified_descent"] and row["exact_directional_derivative"] >= 0 for row in group),
            "all_available_bounds_cover": all(row["directional_bound_covers"] for row in group if row["available"]),
            "equal_parent_gradient_cosine": float(np.mean(cosine_values)) if all(v is not None for v in cosine_values) else None,
            "parent_summaries": parents}
    return summaries


def run_coarse_adjoint_validation(protocol_path: Path, output: Path, *, root: Path | None = None) -> dict:
    """Execute the bounded fixed pilot; callers must obtain the source freeze first."""
    root = Path(__file__).resolve().parents[3] if root is None else root.resolve()
    output = output.resolve()
    if output == root or (output.exists() and any(output.iterdir())):
        raise ValueError("output must be a new or empty directory; preserve existing evidence")
    protected = [root / entry["path"] for entry in json.loads(protocol_path.read_bytes())["inputs"].values()]
    if any(path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("output must not contain a pinned input")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    arrays, basis, inputs = load_coarse_adjoint_inputs(protocol, root)
    provenance = environment_record(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    _write_json(output / "input_provenance.json", inputs)
    _write_json(output / "status.json", {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat()})
    aggregate_rows, block_rows, receipts = [], [], []
    offline = _physics()
    rtol, atol = protocol["numerical_check_relative_tolerance"], protocol["numerical_check_absolute_tolerance"]
    try:
        receivers = tuple(int(i) for i in arrays["receiver_indices"])
        teacher_checks = []
        for case, (parent, slice_id) in enumerate(_IDENTITIES):
            model = arrays["models"][case].astype(np.float64).ravel()
            selected = np.flatnonzero(arrays["block_case_indices"] == case)
            blocks = tuple(SurveyBlock(int(arrays["source_indices"][j]), float(arrays["frequencies_hz"][j]),
                arrays["observations"][j, :, 0].astype(np.float64)+1j*arrays["observations"][j, :, 1].astype(np.float64),
                f"{slice_id}:{int(j)}") for j in selected)
            exact = []
            for block in blocks:
                state = offline.solve_state(model, offline.source(block.source_index), block.frequency_hz)
                residual = state[list(receivers)]-block.observed
                adjoint = offline._solve(model, offline.inject_receivers(residual, receivers),
                    block.frequency_hz, category="adjoint_rhs", adjoint=True)
                gradient = (2*np.pi*block.frequency_hz)**2*np.real(np.conjugate(adjoint)*state)
                exact.append((state, adjoint, gradient, .5*float(np.vdot(residual, residual).real)))
            exact_gradient = np.sum([row[2] for row in exact], axis=0)
            exact_objective = float(sum(row[3] for row in exact))
            gradient_error = float(np.linalg.norm(exact_gradient-arrays["exact_gradients"][case]))
            objective_error = abs(exact_objective-float(arrays["exact_objectives"][case]))
            teacher_checks.append({"slice_id": slice_id, "gradient_absolute_l2_error": gradient_error,
                "gradient_relative_l2_error": gradient_error/max(float(np.linalg.norm(exact_gradient)), 1e-300),
                "objective_absolute_error": objective_error,
                "passes": bool(gradient_error <= atol+rtol*np.linalg.norm(exact_gradient)
                               and objective_error <= atol+rtol*abs(exact_objective))})
            for method in METHODS:
                # Every case/method starts from independent cold caches. Setup
                # is separately reported, never silently shared across methods.
                physics = _physics()
                surrogate = PODGalerkinSurrogate(physics, basis)
                engine = FastCoarseStability(physics, 96)
                setup = {"pod_operator_projection": surrogate.offline_projection_cost,
                         "coarse_stability": engine.cost_snapshot()}
                certifier = PODCoarseAdjointCertifier(physics, surrogate, engine)
                result = certifier.certify_survey(model, blocks, receivers, method=method)
                exact_dd = float(exact_gradient @ result.direction)
                error = _finite(abs(exact_dd-result.approximate_directional_derivative)) if result.approximate_directional_derivative is not None else None
                covers = None if not result.available else _within(error, result.directional_error_bound, rtol, atol)
                delta = result.costs
                zero_fine = all(value == 0 for value in delta["reference_solver"].values()) and all(
                    delta["coarse_stability"][key] == 0 for key in ("fine_grid_factorizations", "fine_grid_rhs", "fine_grid_dense_svd"))
                receipt = {"parent_id": parent, "slice_id": slice_id, "method": method,
                    "setup": setup, "online": delta,
                    "cumulative_coarse_stability": engine.cost_snapshot(),
                    "cumulative_pod": surrogate.cost_snapshot(), "actual_fine_online_work_zero": zero_fine,
                    "model_sha256": result.model_sha256,
                    "stability_contexts": result.stability_contexts}
                receipts.append(receipt)
                gradient_norm = None if result.approximate_gradient is None else float(np.linalg.norm(result.approximate_gradient))
                aggregate_rows.append({"parent_id": parent, "slice_id": slice_id, "method": method,
                    "available": result.available, "certified_descent": result.certified_descent,
                    "exact_fallback_required": result.exact_fallback_required, "reason": result.reason,
                    "exact_objective": exact_objective, "approximate_objective": result.approximate_objective,
                    "exact_gradient_norm": float(np.linalg.norm(exact_gradient)), "approximate_gradient_norm": gradient_norm,
                    "gradient_cosine": _cosine(exact_gradient, result.approximate_gradient),
                    "direction_norm": float(np.linalg.norm(result.direction)), "exact_directional_derivative": exact_dd,
                    "approximate_directional_derivative": result.approximate_directional_derivative,
                    "actual_directional_error": error, "directional_error_bound": result.directional_error_bound,
                    "certified_upper_derivative": result.certified_upper_derivative, "directional_bound_covers": covers,
                    "bound_to_approximate_descent": _ratio(result.directional_error_bound, -result.approximate_directional_derivative)
                        if result.available and result.approximate_directional_derivative < 0 else None,
                    "bound_effectivity": _ratio(result.directional_error_bound, error) if result.available and error is not None and error > 0 else None,
                    "actual_fine_online_work_zero": zero_fine})
                for receipt_block, reference in zip(result.block_certificates, exact, strict=True):
                    block_dd = float(reference[2] @ result.direction)
                    block_error = _finite(abs(block_dd-receipt_block.approximate_directional_derivative)) if receipt_block.approximate_directional_derivative is not None else None
                    block_rows.append({"parent_id": parent, "slice_id": slice_id, "method": method,
                        **receipt_block.to_dict(), "exact_objective": reference[3],
                        "exact_gradient_norm": float(np.linalg.norm(reference[2])),
                        "exact_directional_derivative": block_dd, "actual_directional_error": block_error,
                        "directional_bound_covers": None if not receipt_block.available else _within(
                            block_error, receipt_block.directional_error_bound, rtol, atol),
                        "actual_state_error_l2": float(np.linalg.norm(reference[0]-receipt_block.state)),
                        "actual_adjoint_error_l2": float(np.linalg.norm(reference[1]-receipt_block.adjoint)),
                        "state_error_bound_l2": None if receipt_block.eta_u_components is None else float(np.linalg.norm(receipt_block.eta_u_components)),
                        "adjoint_error_bound_l2": None if receipt_block.eta_lambda_components is None else float(np.linalg.norm(receipt_block.eta_lambda_components))})
                _write_csv(output / "aggregate_cases.csv", aggregate_rows)
                _write_csv(output / "block_cases.csv", block_rows)
                _write_json(output / "cost_receipts.json", receipts)
        _, _, end_inputs = load_coarse_adjoint_inputs(protocol, root)
        unchanged = source_digest(root) == provenance["source_sha256"] and protocol_path.read_bytes() == protocol_bytes
        inputs_unchanged = inputs["verified_file_sha256"] == end_inputs["verified_file_sha256"]
        checks = {
            "complete_fixed_cohort": len(aggregate_rows) == 24 and len(block_rows) == 576,
            "source_unchanged": unchanged, "inputs_unchanged": inputs_unchanged,
            "teacher_aggregate_agreement": all(row["passes"] for row in teacher_checks),
            "fine_online_work_zero": all(row["actual_fine_online_work_zero"] for row in aggregate_rows),
            "available_aggregate_bounds_cover": all(row["directional_bound_covers"] for row in aggregate_rows if row["available"]),
            "available_block_bounds_cover": all(row["directional_bound_covers"] for row in block_rows if row["available"]),
            "no_false_certification": all(not row["certified_descent"] or row["exact_directional_derivative"] < 0 for row in aggregate_rows),
        }
        summary = {"schema_version": 1, "status": "complete" if all(checks.values()) else "checks-failed",
            "study_role": protocol["study_role"], "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "checks": checks, "methods": _summarize(aggregate_rows), "teacher_checks": teacher_checks,
            "provenance": provenance, "source_sha256_at_end": source_digest(root),
            "offline_fine_validation_costs": offline.cost_snapshot(),
            "cost_scope": "Cold standalone setup per case/method; shared historical teacher generation and basis fit retained in input_provenance; all executed online counters in cost_receipts; offline fine validation on a separate solver. No timings or full-inversion speedup claim.",
            "uncertainty_scope": "Descriptive eight correlated slices from four existing validation parents; equal-parent summaries, no IID-eight or held-out inference.",
            "numerical_scope": "Exact-arithmetic deterministic inequalities evaluated in ordinary float64/complex128, with unverified cancellation allowances; no formal roundoff enclosure or finite-step guarantee.",
            "timing_claims": False, "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False}
        _write_json(output / "summary.json", summary)
        _write_json(output / "status.json", {"status": summary["status"], "recorded_utc": datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as exc:
        _write_json(output / "status.json", {"status": "failed", "exception": type(exc).__name__, "message": str(exc),
            "completed_aggregate_rows": len(aggregate_rows), "completed_block_rows": len(block_rows),
            "recorded_utc": datetime.now(timezone.utc).isoformat()})
        raise
