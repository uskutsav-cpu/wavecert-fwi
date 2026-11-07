"""Bounded cold-query cost study on an existing development validation cohort.

The three tasks differ: exact objective/full gradient, raw POD objective/gradient,
and corrected physical proposal plus directional certificate. No time-to-target,
inversion, fresh-geology, or field-scale claim is supported by this study.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import get_clock_info, perf_counter

import numpy as np

from wavecert.certificates.coarse_adjoint import PODCoarseAdjointCertifier
from wavecert.experiments.coarse_adjoint_validation import (
    _cosine,
    _ratio,
    _within,
    load_coarse_adjoint_inputs,
)
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.physics.residual_coded import best_exact_objective_gradient
from wavecert.provenance import environment_record, source_digest
from wavecert.surrogates.pod import PODGalerkinSurrogate

METHODS = ("best_exact_objective_gradient", "raw_POD_objective_gradient", "corrected_POD_directional_certificate")
_THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def validate_runtime_protocol(protocol):
    fixed = {
        "schema_version": 1, "study_role": "existing_validation_cold_runtime_controls",
        "methods": list(METHODS), "repetitions": 3, "pod_rank": 64, "spectral_coarse_rank": 96,
        "method_order": "rotate METHODS by (repetition + case_index) modulo 3; repetition outermost",
        "cache_rule": "new independent physics/POD/coarse-engine/certifier objects for every method/case/repetition; no warm diagnostic",
        "timing_scope": "single-process cold per-model solver caches, not process/OS/CPU-cache cold; no concurrent task-owned compute",
        "thread_environment": dict.fromkeys(_THREAD_KEYS, "1"),
        "numerical_check_relative_tolerance": 1e-9, "numerical_check_absolute_tolerance": 1e-10,
        "heldout_arrays_loaded": False, "inversion_or_time_to_target_claim": False,
    }
    for key, expected in fixed.items():
        if protocol.get(key) != expected:
            raise ValueError(f"fixed runtime protocol mismatch: {key}")
    entry = protocol.get("validation_protocol", {})
    path, digest = entry.get("path", ""), entry.get("sha256", "")
    if (not path or Path(path).is_absolute() or ".." in Path(path).parts
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("a repository-relative exact validation protocol must be pinned")


def load_runtime_inputs(protocol, root):
    """Input-only audit; uses the unchanged fixed development-cohort loader."""
    validate_runtime_protocol(protocol)
    entry = protocol["validation_protocol"]
    raw = (root / entry["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("validation protocol SHA-256 mismatch")
    validation_protocol = json.loads(raw)
    arrays, basis, audit = load_coarse_adjoint_inputs(validation_protocol, root)
    audit = {**audit, "validation_protocol_sha256": entry["sha256"],
             "validation_protocol": validation_protocol}
    return arrays, basis, audit


def _new_physics(config):
    return Helmholtz2D(**{**config, "shape": tuple(config["shape"])})


def _measure_control(method, *, physics_config, basis, model, blocks, receivers,
                     coarse_rank, clock=perf_counter):
    """One cold public-API invocation; injectable clock permits untimed unit tests."""
    if method not in METHODS:
        raise ValueError("unknown runtime control")
    started = clock()
    physics = _new_physics(physics_config)
    surrogate = engine = certifier = None
    if method != METHODS[0]:
        surrogate = PODGalerkinSurrogate(physics, basis)
    if method == METHODS[2]:
        engine = FastCoarseStability(physics, coarse_rank)
        certifier = PODCoarseAdjointCertifier(physics, surrogate, engine)
    ready = clock()
    certificate = None
    strategies, expected_rhs = {}, None
    if method == METHODS[0]:
        exact = best_exact_objective_gradient(physics=physics, m=model, blocks=blocks,
            receiver_indices=receivers)
        objective, gradient = exact.objective, exact.gradient
        strategies = {str(frequency): route for frequency, route in exact.frequency_strategies.items()}
        expected_rhs = exact.expected_rhs
    elif method == METHODS[1]:
        objective, gradient = 0.0, np.zeros(physics.n)
        for block in blocks:
            value, contribution, _, _ = surrogate.objective_and_gradient(model,
                physics.source(block.source_index), block.observed, receivers, block.frequency_hz)
            objective += value
            gradient += contribution
    else:
        certificate = certifier.certify_survey(model, blocks, receivers,
            method="coarse_corrected_oriented")
        objective, gradient = certificate.approximate_objective, certificate.approximate_gradient
    finished = clock()
    # All receipts/quality checks below are outside the timed public query.
    if certificate is not None:
        query_costs = certificate.costs
        cumulative = certifier.cost_snapshot()
        coarse_setup = {key: value-query_costs["coarse_stability"][key]
                        for key, value in cumulative["coarse_stability"].items()}
    else:
        query_costs = {"reference_solver": physics.cost_snapshot(),
                       "pod": None if surrogate is None else surrogate.cost_snapshot()}
        cumulative, coarse_setup = query_costs, None
    setup_costs = {"pod_operator_projection": None if surrogate is None else surrogate.offline_projection_cost,
                   "pod_basis_orthogonality_gram_evaluations": 0 if surrogate is None else 1,
                   "coarse_stability": coarse_setup}
    if method == METHODS[0]:
        executed_rhs = sum(value for key, value in query_costs["reference_solver"].items() if key.endswith("_rhs"))
        if executed_rhs != expected_rhs:
            raise AssertionError("strongest implemented exact route RHS receipt differs from its prediction")
    return {"method": method, "objective": objective, "gradient": gradient, "certificate": certificate,
        "setup_seconds": float(ready-started), "query_seconds": float(finished-ready),
        "total_cold_seconds": float(finished-started), "setup_costs": setup_costs,
        "query_costs": query_costs, "cumulative_costs": cumulative,
        "exact_frequency_strategies": strategies, "exact_expected_rhs": expected_rhs}


def _parity(objective, gradient, previous_objective, previous_gradient, rtol, atol):
    same_objective = (objective is None and previous_objective is None) or (
        objective is not None and previous_objective is not None
        and np.isclose(objective, previous_objective, rtol=rtol, atol=atol))
    same_gradient = (gradient is None and previous_gradient is None) or (
        gradient is not None and previous_gradient is not None
        and np.allclose(gradient, previous_gradient, rtol=rtol, atol=atol))
    return bool(same_objective and same_gradient)


def _cases(arrays):
    cases = []
    receivers = tuple(int(i) for i in arrays["receiver_indices"])
    for i, (parent, slice_id) in enumerate(zip(arrays["parent_ids"], arrays["slice_ids"], strict=True)):
        selected = np.flatnonzero(arrays["block_case_indices"] == i)
        blocks = tuple(SurveyBlock(int(arrays["source_indices"][j]), float(arrays["frequencies_hz"][j]),
            arrays["observations"][j, :, 0].astype(np.float64)+1j*arrays["observations"][j, :, 1].astype(np.float64),
            f"{slice_id}:{int(j)}") for j in selected)
        cases.append({"case_index": i, "parent_id": str(parent), "slice_id": str(slice_id),
                      "model": arrays["models"][i].astype(np.float64).ravel(), "blocks": blocks,
                      "receivers": receivers})
    return cases


def _offline_references(cases, config, arrays, rtol, atol):
    physics = _new_physics(config)
    references = []
    for case in cases:
        objective, gradient, blocks = 0.0, np.zeros(physics.n), []
        for block in case["blocks"]:
            value, contribution, _, _ = physics.objective_and_gradient(case["model"],
                physics.source(block.source_index), block.observed, case["receivers"], block.frequency_hz)
            objective += value
            gradient += contribution
            blocks.append({"objective": value, "gradient": contribution})
        index = case["case_index"]
        gerr = float(np.linalg.norm(gradient-arrays["exact_gradients"][index]))
        oerr = float(abs(objective-arrays["exact_objectives"][index]))
        references.append({"objective": objective, "gradient": gradient, "blocks": blocks,
            "stored_teacher_gradient_absolute_error": gerr, "stored_teacher_objective_absolute_error": oerr,
            "stored_teacher_agreement": bool(gerr <= atol+rtol*np.linalg.norm(gradient)
                and oerr <= atol+rtol*abs(objective))})
    return references, physics.cost_snapshot()


def _json(path, record):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, tuple, dict)) else value
                         for key, value in row.items()} for row in rows)


def _method_order(repetition, case_index):
    shift = (repetition+case_index) % len(METHODS)
    return METHODS[shift:]+METHODS[:shift]


def _paired(rows):
    pairs = []
    for repetition in range(3):
        for case in range(8):
            group = {row["method"]: row for row in rows
                     if row["repetition"] == repetition and row["case_index"] == case}
            if not group:
                continue
            if set(group) != set(METHODS):
                raise ValueError("cannot export an incomplete three-method timing pair")
            row = group[METHODS[0]]
            pairs.append({"repetition": repetition, "case_index": case,
                "parent_id": row["parent_id"], "slice_id": row["slice_id"],
                **{f"{method}_{key}": group[method][key] for method in METHODS
                   for key in ("setup_seconds", "query_seconds", "total_cold_seconds")}})
    return pairs


def _describe(rows):
    result = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        parents = {}
        for parent in sorted({row["parent_id"] for row in selected}):
            group = [row for row in selected if row["parent_id"] == parent]
            parents[parent] = {"n_slices": len({row["slice_id"] for row in group}), "n_calls": len(group),
                **{f"{key}_median": float(np.median([row[key] for row in group]))
                   for key in ("setup_seconds", "query_seconds", "total_cold_seconds")},
                "gradient_cosine_mean": float(np.mean([row["gradient_cosine"] for row in group]))
                    if all(row["gradient_cosine"] is not None for row in group) else None,
                "certified_calls": sum(row["certified_descent"] is True for row in group)}
        result[method] = {"n_calls": len(selected), "parents": parents,
            **{f"{key}_median": float(np.median([row[key] for row in selected]))
               for key in ("setup_seconds", "query_seconds", "total_cold_seconds")},
            "equal_parent_mean_total_cold_seconds": float(np.mean([
                np.mean([row["total_cold_seconds"] for row in selected if row["parent_id"] == parent])
                for parent in parents]))}
    return result


def run_runtime_controls(protocol_path, output, *, root=None, compute_context="unspecified"):
    """Execute only in a deliberately idle compute session; no concurrency lock is inferred."""
    root = Path(__file__).resolve().parents[3] if root is None else root.resolve()
    output = output.resolve()
    if output == root or (output.exists() and any(output.iterdir())):
        raise ValueError("runtime output must be new or empty; preserve existing evidence")
    io_start = perf_counter()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    arrays, basis, inputs = load_runtime_inputs(protocol, root)
    input_seconds = perf_counter()-io_start
    if any(os.environ.get(key) != "1" for key in _THREAD_KEYS):
        raise ValueError("set all five protocol thread-environment variables to 1 before starting Python")
    if protocol_path.resolve().is_relative_to(output) or any(
            (root / item["path"]).resolve().is_relative_to(output)
            for item in inputs["validation_protocol"]["inputs"].values()):
        raise ValueError("runtime output must not contain pinned inputs")
    provenance = environment_record(root)
    load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    start_preparation = perf_counter()
    cases = _cases(arrays)
    preparation_seconds = perf_counter()-start_preparation
    rtol, atol = protocol["numerical_check_relative_tolerance"], protocol["numerical_check_absolute_tolerance"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    _json(output / "input_provenance.json", inputs)
    _json(output / "status.json", {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat()})
    rows, block_rows, receipts, first_results = [], [], [], {}
    try:
        reference_start = perf_counter()
        references, offline_costs = _offline_references(cases, inputs["validation_protocol"]["physics"], arrays, rtol, atol)
        reference_seconds = perf_counter()-reference_start
        load_at_timing_start = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        for repetition in range(3):
            for case in cases:
                reference = references[case["case_index"]]
                for position, method in enumerate(_method_order(repetition, case["case_index"])):
                    result = _measure_control(method, physics_config=inputs["validation_protocol"]["physics"],
                        basis=basis, model=case["model"], blocks=case["blocks"], receivers=case["receivers"], coarse_rank=96)
                    gradient, certificate = result["gradient"], result["certificate"]
                    norm = None if gradient is None else float(np.linalg.norm(gradient))
                    finite_proposal = bool(result["objective"] is not None and np.isfinite(result["objective"])
                        and gradient is not None and np.all(np.isfinite(gradient))
                        and norm is not None and np.isfinite(norm))
                    normalized = (-gradient/norm) if norm is not None and np.isfinite(norm) and norm > 1e-14 else np.zeros(576)
                    direction = normalized if certificate is None else certificate.direction
                    exact_dd = float(reference["gradient"] @ direction)
                    key = (case["case_index"], method)
                    if key not in first_results:
                        first_results[key] = (result["objective"], gradient)
                    previous_objective, previous_gradient = first_results[key]
                    repeat_parity = _parity(result["objective"], gradient, previous_objective, previous_gradient, rtol, atol)
                    objective_error = None if result["objective"] is None else float(abs(result["objective"]-reference["objective"]))
                    gradient_error = None if gradient is None else float(np.linalg.norm(gradient-reference["gradient"]))
                    exact_parity = None if method != METHODS[0] else bool(
                        objective_error <= atol+rtol*abs(reference["objective"])
                        and gradient_error <= atol+rtol*np.linalg.norm(reference["gradient"]))
                    error, covers, certificate_parity = None, None, None
                    if certificate is not None:
                        center = certificate.approximate_directional_derivative
                        error = None if center is None else float(abs(exact_dd-center))
                        covers = None if not certificate.available else _within(error, certificate.directional_error_bound, rtol, atol)
                        certificate_parity = bool(np.allclose(normalized, certificate.direction, rtol=rtol, atol=atol)
                            and (gradient is None or np.allclose(gradient,
                                np.sum([b.physical_gradient for b in certificate.block_certificates], axis=0), rtol=rtol, atol=atol))
                            and (center is None or np.isclose(center, gradient @ direction, rtol=rtol, atol=atol)))
                        for block, fine in zip(certificate.block_certificates, reference["blocks"], strict=True):
                            derivative = float(fine["gradient"] @ direction)
                            block_error = None if block.approximate_directional_derivative is None else float(abs(derivative-block.approximate_directional_derivative))
                            block_rows.append({"repetition": repetition, "case_index": case["case_index"],
                                "parent_id": case["parent_id"], "slice_id": case["slice_id"], **block.to_dict(),
                                "exact_directional_derivative": derivative, "actual_directional_error": block_error,
                                "directional_bound_covers": None if not block.available else _within(block_error, block.directional_error_bound, rtol, atol)})
                    receipt = {"repetition": repetition, "case_index": case["case_index"], "parent_id": case["parent_id"],
                        "slice_id": case["slice_id"], "method": method,
                        **{k: result[k] for k in ("setup_costs", "query_costs", "cumulative_costs", "exact_frequency_strategies", "exact_expected_rhs")},
                        "certificate_contexts": None if certificate is None else certificate.stability_contexts,
                        "model_sha256": None if certificate is None else certificate.model_sha256}
                    fine_costs = result["query_costs"]["reference_solver"]
                    approximation_fine_zero = None if method == METHODS[0] else all(value == 0 for value in fine_costs.values())
                    if certificate is not None:
                        approximation_fine_zero = approximation_fine_zero and all(
                            result["query_costs"]["coarse_stability"][key] == 0
                            for key in ("fine_grid_factorizations", "fine_grid_rhs", "fine_grid_dense_svd"))
                    receipts.append(receipt)
                    rows.append({"repetition": repetition, "case_index": case["case_index"], "parent_id": case["parent_id"],
                        "slice_id": case["slice_id"], "method": method, "order_position": position,
                        **{key: result[key] for key in ("setup_seconds", "query_seconds", "total_cold_seconds")},
                        "objective": result["objective"], "reference_objective": reference["objective"],
                        "objective_absolute_error": objective_error, "gradient_norm": norm,
                        "reference_gradient_norm": float(np.linalg.norm(reference["gradient"])),
                        "gradient_relative_l2_error": None if gradient_error is None else _ratio(gradient_error, float(np.linalg.norm(reference["gradient"]))),
                        "gradient_cosine": _cosine(gradient, reference["gradient"]), "exact_directional_derivative": exact_dd,
                        "true_descent": exact_dd < 0, "repeat_parity": repeat_parity, "exact_gradient_objective_parity": exact_parity,
                        "finite_objective_gradient": finite_proposal,
                        "certificate_center_gradient_direction_parity": certificate_parity,
                        "actual_approximation_fine_solve_work_zero": approximation_fine_zero,
                        "certificate_available": None if certificate is None else certificate.available,
                        "certified_descent": None if certificate is None else certificate.certified_descent,
                        "exact_fallback_required": None if certificate is None else certificate.exact_fallback_required,
                        "approximate_directional_derivative": None if certificate is None else certificate.approximate_directional_derivative,
                        "directional_error_bound": None if certificate is None else certificate.directional_error_bound,
                        "certified_upper_derivative": None if certificate is None else certificate.certified_upper_derivative,
                        "actual_directional_error": error, "directional_bound_covers": covers})
                # Keep output I/O outside complete rotated triples.
                _csv(output / "cold_calls.csv", rows)
                _csv(output / "paired_cold_calls.csv", _paired(rows))
                _csv(output / "certificate_blocks.csv", block_rows)
                _json(output / "cost_receipts.json", receipts)
        _, _, end_inputs = load_runtime_inputs(protocol, root)
        unchanged = source_digest(root) == provenance["source_sha256"] and protocol_path.read_bytes() == protocol_bytes
        checks = {
            "complete_fixed_cohort": len(rows) == 72 and len(block_rows) == 576,
            "source_unchanged": unchanged,
            "inputs_unchanged": inputs["verified_file_sha256"] == end_inputs["verified_file_sha256"],
            "stored_teacher_reference_agreement": all(row["stored_teacher_agreement"] for row in references),
            "finite_objectives_and_gradients": all(row["finite_objective_gradient"] for row in rows),
            "all_cold_repeat_parities": all(row["repeat_parity"] for row in rows),
            "exact_reference_parity": all(row["exact_gradient_objective_parity"] for row in rows if row["method"] == METHODS[0]),
            "certificate_gradient_direction_parity": all(row["certificate_center_gradient_direction_parity"] for row in rows if row["method"] == METHODS[2]),
            "approximation_fine_solve_work_zero": all(row["actual_approximation_fine_solve_work_zero"] for row in rows if row["method"] != METHODS[0]),
            "available_aggregate_bounds_cover": all(row["directional_bound_covers"] for row in rows if row["certificate_available"]),
            "available_block_bounds_cover": all(row["directional_bound_covers"] for row in block_rows if row["available"]),
            "no_false_certifications": all(not row["certified_descent"] or row["true_descent"] for row in rows),
        }
        summary = {"schema_version": 1, "status": "complete" if all(checks.values()) else "checks-failed",
            "study_role": protocol["study_role"], "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "checks": checks, "descriptive_methods": _describe(rows), "provenance": provenance,
            "source_sha256_at_end": source_digest(root), "common_input_io_and_audit_seconds": input_seconds,
            "common_case_preparation_seconds": preparation_seconds,
            "offline_fine_validation_seconds": reference_seconds, "offline_fine_validation_costs": offline_costs,
            "teacher_reference_checks": [{key: value for key, value in ref.items() if key not in {"gradient", "blocks"}}
                                         for ref in references],
            "clock": vars(get_clock_info("perf_counter")), "compute_context": compute_context,
            "load_average_before": load_before, "load_average_after": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "load_average_at_timing_start": load_at_timing_start,
            "scope": "72 cold per-model calls on eight existing correlated validation slices/four parents; three repeated visits with new caches. Tasks differ: exact objective/full gradient, raw POD objective/gradient, corrected physical proposal+DD certificate. Descriptive paired timings only, no IID-eight CI, process-cold, field-scale, inversion, or time-to-target claim. Shared teacher/basis development costs retained in input_provenance."}
        _json(output / "summary.json", summary)
        _json(output / "status.json", {"status": summary["status"], "recorded_utc": datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as exc:
        _csv(output / "cold_calls.csv", rows)
        _csv(output / "certificate_blocks.csv", block_rows)
        _json(output / "cost_receipts.json", receipts)
        _json(output / "status.json", {"status": "failed", "exception": type(exc).__name__, "message": str(exc),
            "completed_cold_calls": len(rows), "recorded_utc": datetime.now(timezone.utc).isoformat()})
        raise
