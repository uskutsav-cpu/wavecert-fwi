"""Fixed validation-only batching development; parity precedes comparative time.

This is a post-confirmation engineering extension, not a new held-out study.
Existing production modules are hash-pinned and never modified by this runner.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np

from wavecert.certificates.coarse_adjoint import PODCoarseAdjointCertifier
from wavecert.experiments.coarse_adjoint_validation import (
    _cosine,
    _within,
    load_coarse_adjoint_inputs,
)
from wavecert.experiments.coarse_runtime_controls import _cases, _offline_references, _parity
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.physics.residual_coded import best_exact_objective_gradient
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.pod import PODGalerkinSurrogate

PROTOCOL_SHA256 = "a53600eff96f128193f813c077da29235d185fca473a24e060ca8fe75334ad1f"
METHODS = ("best_exact", "raw_POD_scalar", "raw_POD_batch", "corrected_scalar96",
           "corrected_batch96", "corrected_batch64")
CANDIDATES = ("corrected_batch96", "corrected_batch64")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
EXECUTION_MODULES = ("wavecert.experiments.batched_pod_development", "wavecert.experiments.coarse_runtime_controls",
    "wavecert.experiments.coarse_adjoint_validation", "wavecert.surrogates.pod_batch",
    "wavecert.physics.coarse_batch", "wavecert.certificates.coarse_adjoint_batch")
COST_UNITS = {"fine_and_reduced_and_coarse_rhs": "actual matrix RHS columns; dimensions differ",
    "coarse_projection_calls_and_coarse_lift_calls": "logical single-vector columns, including batched execution",
    "pod_projections_and_lifts": "logical single-vector columns",
    "coarse_solve_calls": "actual reduced LU solve dispatches; separate from RHS columns",
    "dst_I_real_transforms_and_idst_I_real_transforms": "real-equivalent spatial transforms per column; complex counts twice",
    "batch_dispatches": "actual grouped matrix/FFT/stencil/norm API calls; not RHS columns or FLOPs"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()} for row in rows)


def _delta(before, after):
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in set(before) | set(after)
            if key not in {"full_dimension", "reduced_dimension"}}


def validate_protocol(protocol):
    _require(protocol["study_role"] == "post_confirmation_validation_only_batched_POD_development"
             and protocol["methods"] == list(METHODS), "fixed development role/methods changed")
    _require(protocol["cohort"]["n_slices"] == 8 and protocol["cohort"]["n_parents"] == 4
             and protocol["cohort"]["blocks_per_slice"] == 24 and protocol["cohort"]["pod_rank"] == 64,
             "fixed validation cohort/POD rank changed")
    _require(protocol["timing"]["repetitions"] == 6 and protocol["timing"]["n_cold_calls"] == 288
             and protocol["candidate_rule"]["candidate_labels"] == list(CANDIDATES), "fixed cost/rank design changed")
    _require(protocol["calibration_or_test_arrays_loaded"] is False
             and protocol["external_test_arrays_loaded"] is False, "development cannot load test arrays")


def load_development_inputs(protocol_path, root):
    """Input-only audit: no surrogate construction, predictions, or timings."""
    root = Path(root).resolve()
    raw = Path(protocol_path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA256, "fixed development protocol bytes changed")
    protocol = json.loads(raw)
    validate_protocol(protocol)
    pinned = {}
    for name, expected in protocol["unchanged_production_sources"].items():
        path = (root / name).resolve()
        _require(path.is_relative_to(root) and sha256_file(path) == expected, "unchanged production source differs")
        module = import_module(name.removeprefix("src/").removesuffix(".py").replace("/", "."))
        _require(Path(module.__file__).resolve() == path, "production module executes from another checkout")
        pinned[name] = expected
    executing = {}
    for name in EXECUTION_MODULES:
        path = root.joinpath("src", *name.split(".")).with_suffix(".py").resolve()
        module = import_module(name)
        _require(Path(module.__file__).resolve() == path, f"execution module from another checkout: {name}")
        executing[path.relative_to(root).as_posix()] = sha256_file(path)
    entry = protocol["validation_protocol"]
    path = (root / entry["path"]).resolve()
    _require(path.is_relative_to(root), "validation protocol escapes root")
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == entry["sha256"], "fixed validation protocol changed")
    arrays, basis, audit = load_coarse_adjoint_inputs(json.loads(payload), root)
    _require(audit["verified_file_sha256"]["basis"] == protocol["cohort"]["basis_sha256"], "POD basis changed")
    return protocol, raw, arrays, basis, {**audit, "unchanged_production_sources": pinned,
                                        "verified_execution_modules": executing,
                                        "validation_protocol_sha256": entry["sha256"]}


def method_order(repetition, case_index):
    shift = (repetition + case_index) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def _new_physics(config):
    return Helmholtz2D(**{**config, "shape": tuple(config["shape"])})


def execute_control(method, *, config, basis, case, clock=None):
    """Cold existing/new API call; clock=None means no timing in parity work.

    Scalar64 is an additional untimed parity reference only. Returned handles
    are used for paid preflight adjoint diagnostics and never serialized.
    """
    _require(method in METHODS or method == "corrected_scalar64", "unknown control")
    start = clock() if clock is not None else None
    physics = _new_physics(config)
    surrogate = engine = certifier = None
    corrected = method.startswith("corrected_")
    if method != "best_exact":
        surrogate = PODGalerkinSurrogate(physics, basis)
    if corrected:
        rank = 64 if method.endswith("64") else 96
        engine = FastCoarseStability(physics, rank)
        if "batch" in method:
            from wavecert.certificates.coarse_adjoint_batch import BatchedPODCoarseAdjointCertifier
            certifier = BatchedPODCoarseAdjointCertifier(physics, surrogate, engine)
        else:
            certifier = PODCoarseAdjointCertifier(physics, surrogate, engine)
    ready = clock() if clock is not None else None
    model, blocks, receivers = case["model"], case["blocks"], case["receivers"]
    certificate = None
    states = adjoints = block_gradients = block_objectives = None
    dispatches, exact_strategies, expected_rhs = {}, {}, None
    if method == "best_exact":
        exact = best_exact_objective_gradient(physics=physics, m=model, blocks=blocks,
                                              receiver_indices=receivers)
        objective, gradient = exact.objective, exact.gradient
        exact_strategies, expected_rhs = exact.frequency_strategies, exact.expected_rhs
    elif method == "raw_POD_scalar":
        values, gradients, fields = [], [], []
        gradient, objective = np.zeros(physics.n), 0.0
        for block in blocks:
            value, contribution, state, _ = surrogate.objective_and_gradient(model,
                physics.source(block.source_index), block.observed, receivers, block.frequency_hz)
            values.append(value)
            gradients.append(contribution)
            fields.append(state)
            gradient += contribution
            objective += value
        block_objectives, block_gradients, states = values, gradients, fields
        dispatches = {"scalar_objective_gradient_API_calls": len(blocks)}
    elif method == "raw_POD_batch":
        from wavecert.surrogates.pod_batch import batched_pod_survey_fields
        result = batched_pod_survey_fields(surrogate, model, blocks, receivers)
        objective, gradient = result.objective, result.aggregate_gradient
        states, adjoints = result.block_states, result.block_adjoints
        block_gradients, block_objectives = result.block_gradients, result.block_objectives
        dispatches = dict(result.batch_dispatches)
    else:
        certificate = certifier.certify_survey(model, blocks, receivers, method="coarse_corrected_oriented")
        objective, gradient = certificate.approximate_objective, certificate.approximate_gradient
    finish = clock() if clock is not None else None
    # Receipts, exported field collection and validation below are outside core time.
    if method == "raw_POD_scalar":
        block_objectives = np.asarray(block_objectives)
        block_gradients, states = np.stack(block_gradients, axis=1), np.stack(states, axis=1)
    if certificate is not None:
        by_key = {(block.source_index, block.frequency_hz): block for block in certificate.block_certificates}
        ordered = [by_key[block.source_index, block.frequency_hz] for block in blocks]
        states = np.stack([block.state for block in ordered], axis=1)
        adjoints = np.stack([block.adjoint for block in ordered], axis=1)
        if all(block.physical_gradient is not None for block in ordered):
            block_gradients = np.stack([block.physical_gradient for block in ordered], axis=1)
        block_objectives = np.asarray([block.approximate_objective for block in ordered])
        costs = certificate.costs
        dispatches = dict(costs.get("batch_dispatches", {}))
    else:
        costs = {"reference_solver": physics.cost_snapshot(),
                 "pod": None if surrogate is None else surrogate.cost_snapshot(),
                 "batch_dispatches": dispatches}
    if expected_rhs is not None:
        _require(sum(value for key, value in physics.cost_snapshot().items() if key.endswith("_rhs")) == expected_rhs,
                 "exact route RHS receipt mismatch")
    cumulative = {"reference_solver": physics.cost_snapshot(),
                  "pod": None if surrogate is None else surrogate.cost_snapshot(),
                  "coarse_stability": None if engine is None else engine.cost_snapshot()}
    coarse_setup = None if engine is None else {
        key: value - costs["coarse_stability"][key] for key, value in cumulative["coarse_stability"].items()}
    setup = {"pod_operator_projection": None if surrogate is None else surrogate.offline_projection_cost,
             "pod_basis_orthogonality_gram_evaluations": int(surrogate is not None),
             "coarse_stability": coarse_setup}
    return dict(method=method, objective=objective, gradient=gradient, certificate=certificate,
        states=states, adjoints=adjoints, block_gradients=block_gradients, block_objectives=block_objectives,
        setup_seconds=None if start is None else ready - start,
        query_seconds=None if start is None else finish - ready,
        total_cold_seconds=None if start is None else finish - start,
        setup_costs=setup, query_costs=costs, cumulative_costs=cumulative, batch_dispatches=dispatches,
        block_keys=[(int(block.source_index), float(block.frequency_hz)) for block in blocks],
        exact_frequency_strategies={str(k): v for k, v in exact_strategies.items()}, exact_expected_rhs=expected_rhs,
        _physics=physics, _surrogate=surrogate)


def _add_paid_scalar_adjoint_fields(result, case):
    """Preflight-only extra adjoints; every additional solve/projection is charged."""
    surrogate, physics = result["_surrogate"], result["_physics"]
    before = surrogate.cost_snapshot()
    adjoints = []
    for i, block in enumerate(case["blocks"]):
        _, _, factor = surrogate._prepare(case["model"], block.frequency_hz)
        residual = result["states"][:, i][list(case["receivers"])] - block.observed
        injection = physics.inject_receivers(residual, case["receivers"])
        surrogate._count("adjoint_rhs_projections")
        coefficients = surrogate._solve(factor, surrogate.basis.T @ injection,
                                         category="reduced_adjoint_rhs", adjoint=True)
        surrogate._count("fine_adjoint_lifts")
        adjoints.append(surrogate.basis @ coefficients)
    result["adjoints"] = np.stack(adjoints, axis=1)
    return dict(pod=_delta(before, surrogate.cost_snapshot()), receiver_injections=len(adjoints),
                scope="Additional untimed raw-scalar adjoint fields for parity only; not hidden in timed controls")


def compare_numeric(first, second, *, rtol, atol):
    if first is None or second is None:
        return dict(passes=first is None and second is None, maximum_absolute_error=None, relative_l2_error=None)
    left, right = np.asarray(first), np.asarray(second)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return dict(passes=False, maximum_absolute_error=None, relative_l2_error=None)
    difference = np.linalg.norm((left - right).ravel())
    scale = max(float(np.linalg.norm(left.ravel())), float(np.linalg.norm(right.ravel())), 1e-300)
    if not np.isfinite(difference) or not np.isfinite(scale):
        return dict(passes=False, maximum_absolute_error=None, relative_l2_error=None)
    return dict(passes=bool(np.allclose(left, right, rtol=rtol, atol=atol)),
                maximum_absolute_error=float(np.max(np.abs(left - right))) if left.size else 0.,
                relative_l2_error=float(difference / scale))


def compare_results(first, second, *, rtol, atol):
    comparisons = {name: compare_numeric(first[name], second[name], rtol=rtol, atol=atol)
                   for name in ("objective", "gradient", "states", "adjoints", "block_gradients", "block_objectives")}
    a, b = first["certificate"], second["certificate"]
    if a is not None or b is not None:
        _require(a is not None and b is not None, "certificate parity needs two certificate results")
        comparisons["direction"] = compare_numeric(a.direction, b.direction, rtol=rtol, atol=atol)
        for name in ("approximate_directional_derivative", "directional_error_bound", "certified_upper_derivative"):
            comparisons[name] = compare_numeric(getattr(a, name), getattr(b, name), rtol=rtol, atol=atol)
        decisions = ("available", "certified_descent", "exact_fallback_required")
        comparisons["aggregate_decisions"] = dict(passes=all(getattr(a, name) == getattr(b, name) for name in decisions))
        comparisons["context_identity"] = dict(passes=a.model_sha256 == b.model_sha256
            and set(a.stability_contexts) == set(b.stability_contexts)
            and all(a.stability_contexts[k] == b.stability_contexts[k] for k in a.stability_contexts))
        aa = {(r.source_index, r.frequency_hz): r for r in a.block_certificates}
        bb = {(r.source_index, r.frequency_hz): r for r in b.block_certificates}
        comparisons["block_keys"] = dict(passes=set(aa) == set(bb))
        for key in sorted(set(aa) & set(bb)):
            left, right = aa[key], bb[key]
            for name in ("approximate_directional_derivative", "directional_error_bound", "primal_residual_components",
                         "adjoint_residual_components", "eta_u_components", "eta_lambda_components", "eta_receiver_state", "bound_terms"):
                comparisons[f"block_{key}_{name}"] = compare_numeric(getattr(left, name), getattr(right, name), rtol=rtol, atol=atol)
            comparisons[f"block_{key}_decisions"] = dict(passes=all(getattr(left, name) == getattr(right, name)
                for name in ("available", "stability_available", "correction_applied", "correction_skipped")))
    return dict(passes=all(value["passes"] for value in comparisons.values()), comparisons=comparisons)


def finite_preflight_fields(result, case):
    """A matching pair of missing/overflowed outputs is never valid parity."""
    n, columns = len(case["model"]), len(case["blocks"])
    shapes = {"objective": (), "gradient": (n,), "states": (n, columns),
              "adjoints": (n, columns), "block_gradients": (n, columns), "block_objectives": (columns,)}
    checks = {name: result[name] is not None and np.asarray(result[name]).shape == shape
              and bool(np.all(np.isfinite(result[name]))) for name, shape in shapes.items()}
    checks["gradient_norm"] = bool(checks["gradient"] and np.isfinite(np.linalg.norm(result["gradient"])))
    return dict(passes=all(checks.values()), fields=checks)


def _numeric_certificate_failure(certificate):
    return bool(certificate is not None and (
        "nonfinite" in (certificate.reason or "")
        or any("nonfinite" in (block.reason or "") for block in certificate.block_certificates)
        or any("nonfinite" in context["reason"] for context in certificate.stability_contexts.values())))


def quality_records(result, reference, *, identity, rtol, atol):
    """All-block quality at one shared direction; never use mean block cosine as primary."""
    gradient, certificate = result["gradient"], result["certificate"]
    norm = None if gradient is None else float(np.linalg.norm(gradient))
    finite = bool(result["objective"] is not None and np.isfinite(result["objective"])
                  and gradient is not None and np.all(np.isfinite(gradient)) and norm is not None and np.isfinite(norm))
    direction = -gradient / norm if finite and norm > 1e-14 else np.zeros_like(reference["gradient"])
    if certificate is not None:
        direction = certificate.direction
    exact_dd = float(reference["gradient"] @ direction)
    center = float(gradient @ direction) if finite else None
    if certificate is not None:
        center = certificate.approximate_directional_derivative
    error = None if center is None else float(abs(exact_dd - center))
    bound = None if certificate is None else certificate.directional_error_bound
    available = None if certificate is None else certificate.available
    covers = None if not available else _within(error, bound, rtol, atol)
    row = {**identity, "objective": result["objective"], "gradient_norm": norm,
        "gradient_cosine": _cosine(gradient, reference["gradient"]), "finite_proposal": finite,
        "direction_norm": float(np.linalg.norm(direction)), "exact_directional_derivative": exact_dd,
        "true_descent": exact_dd < 0, "approximate_directional_derivative": center,
        "actual_directional_error": error, "certificate_available": available,
        "certified_descent": None if certificate is None else certificate.certified_descent,
        "exact_fallback_required": None if certificate is None else certificate.exact_fallback_required,
        "directional_error_bound": bound, "certified_upper_derivative": None if certificate is None else certificate.certified_upper_derivative,
        "directional_bound_covers": covers}
    row["numeric_certificate_failure"] = _numeric_certificate_failure(certificate)
    block_rows = []
    by_key = {} if certificate is None else {
        (int(block.source_index), float(block.frequency_hz)): block for block in certificate.block_certificates}
    ordered_certificates = [] if certificate is None else [by_key[key] for key in result["block_keys"]]
    for i, fine in enumerate(reference["blocks"]):
        contribution = fine["gradient"] if result["method"] == "best_exact" else (
            None if result["block_gradients"] is None else result["block_gradients"][:, i])
        block_center = None if contribution is None else float(contribution @ direction)
        exact_block_dd = float(fine["gradient"] @ direction)
        extra = {} if certificate is None else ordered_certificates[i].to_dict()
        block_bound = None if certificate is None else ordered_certificates[i].directional_error_bound
        block_available = None if certificate is None else ordered_certificates[i].available
        block_error = None if block_center is None else abs(exact_block_dd - block_center)
        source, frequency = result["block_keys"][i]
        block_rows.append({**identity, "block_index": i, "source_index": source, "frequency_hz": frequency,
            "gradient_cosine": _cosine(contribution, fine["gradient"]),
            "exact_directional_derivative": exact_block_dd, "approximate_directional_derivative": block_center,
            "actual_directional_error": block_error, "certificate_available": block_available,
            "directional_error_bound": block_bound,
            "directional_bound_covers": None if not block_available else _within(block_error, block_bound, rtol, atol),
            "certificate_diagnostics": extra,
            "quality_scope": "offline exact block reference; timed exact API returns aggregate" if result["method"] == "best_exact" else "actual proposal block gradient"})
    row["all_block_bounds_cover"] = None if certificate is None else bool(all(
        block["certificate_available"] and block["directional_bound_covers"] for block in block_rows))
    return row, block_rows, direction


def preflight(cases, basis, config, references, *, rtol, atol, output=None):
    records, receipts = [], []
    for case, reference in zip(cases, references, strict=True):
        identity = {key: case[key] for key in ("case_index", "parent_id", "slice_id")}
        for scalar, batch in (("raw_POD_scalar", "raw_POD_batch"),
                              ("corrected_scalar96", "corrected_batch96"),
                              ("corrected_scalar64", "corrected_batch64")):
            left = execute_control(scalar, config=config, basis=basis, case=case)
            extra = _add_paid_scalar_adjoint_fields(left, case) if scalar == "raw_POD_scalar" else None
            right = execute_control(batch, config=config, basis=basis, case=case)
            comparison = compare_results(left, right, rtol=rtol, atol=atol)
            left_fields, right_fields = finite_preflight_fields(left, case), finite_preflight_fields(right, case)
            comparison["finite_fields"] = {"scalar": left_fields, "batch": right_fields}
            comparison["passes"] = bool(comparison["passes"] and left_fields["passes"] and right_fields["passes"]
                and not _numeric_certificate_failure(left["certificate"])
                and not _numeric_certificate_failure(right["certificate"]))
            quality, _, _ = quality_records(right, reference, identity={**identity, "method": batch}, rtol=rtol, atol=atol)
            records.append({**identity, "comparison": f"{scalar}_vs_{batch}", **comparison, "batch_quality": quality})
            for result in (left, right):
                receipts.append({**identity, "method": result["method"], "setup": result["setup_costs"],
                    "query": result["query_costs"], "batch_dispatches": result["batch_dispatches"],
                    "cost_units": COST_UNITS,
                    "extra_scalar_adjoint_parity_work": extra if result is left else None})
            if output is not None:
                _json(output / "parity_checks.json", records)
                _json(output / "parity_work_receipts.json", receipts)
    _require(len(records) == 24 and all(row["passes"] for row in records),
             "finite complete fields and all eight raw/rank96/rank64 parity checks are required before comparative timings")
    return records, receipts


def paired_rows(rows):
    paired = []
    for repetition in range(6):
        for index in range(8):
            group = {r["method"]: r for r in rows if r["repetition"] == repetition and r["case_index"] == index}
            if not group:
                continue
            _require(set(group) == set(METHODS), "incomplete six-method timing pair")
            base = group[METHODS[0]]
            paired.append({"repetition": repetition, "case_index": index, "parent_id": base["parent_id"],
                "slice_id": base["slice_id"], **{f"{m}_{key}": group[m][key] for m in METHODS
                    for key in ("setup_seconds", "query_seconds", "total_cold_seconds")}})
    return paired


def describe_methods(rows):
    """Six visits measure costs; they do not multiply geological sample size."""
    descriptions = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        parents = {}
        for parent in sorted({row["parent_id"] for row in selected}):
            parent_rows = [row for row in selected if row["parent_id"] == parent]
            cases = sorted({row["case_index"] for row in parent_rows})
            first = [row for row in parent_rows if row["repetition"] == 0]
            parent_result = {"n_slices": len(cases), "n_cold_visits": len(parent_rows)}
            for key in ("setup_seconds", "query_seconds", "total_cold_seconds"):
                medians = [float(np.median([row[key] for row in parent_rows if row["case_index"] == index]))
                           for index in cases]
                parent_result[f"mean_slice_median_{key}"] = float(np.mean(medians))
            parent_result["mean_slice_gradient_cosine"] = (float(np.mean([row["gradient_cosine"] for row in first]))
                if len(first) == len(cases) and all(row["gradient_cosine"] is not None for row in first) else None)
            parents[parent] = parent_result
        descriptions[method] = {"n_cold_visits": len(selected), "n_parents": len(parents),
            "parents": parents, "equal_parent_mean_slice_median_total_cold_seconds":
                float(np.mean([p["mean_slice_median_total_cold_seconds"] for p in parents.values()])) if parents else None}
    return descriptions


def candidate_recommendation(rows, parity_records):
    """Fixed development-only eligibility and equal-parent cold-cost rule."""
    candidates = {}
    for method in CANDIDATES:
        selected = [r for r in rows if r["method"] == method]
        pre = [r for r in parity_records if r["batch_quality"]["method"] == method]
        complete = (len(selected) == 48 and {(r["repetition"], r["case_index"]) for r in selected}
                    == {(rep, index) for rep in range(6) for index in range(8)} and len(pre) == 8)
        eligible = complete and all(r["passes"] and r["batch_quality"]["certified_descent"]
            and r["batch_quality"]["directional_bound_covers"] and r["batch_quality"]["all_block_bounds_cover"] for r in pre)
        eligible = bool(eligible and all(r["finite_proposal"] and r["repeat_parity"] and r["certified_descent"]
            and r["certificate_available"] and r["directional_bound_covers"] and r["all_block_bounds_cover"] for r in selected))
        parent_medians = {}
        if complete:
            for parent in sorted({r["parent_id"] for r in selected}):
                parent_rows = [r for r in selected if r["parent_id"] == parent]
                medians = [float(np.median([r["total_cold_seconds"] for r in parent_rows if r["case_index"] == index]))
                           for index in sorted({r["case_index"] for r in parent_rows})]
                parent_medians[parent] = float(np.mean(medians))
        score = float(np.mean(list(parent_medians.values()))) if parent_medians else None
        candidates[method] = dict(eligible=eligible, complete=complete, equal_parent_mean_slice_median_total_cold_seconds=score,
                                   parent_means_of_slice_medians=parent_medians)
    eligible = [label for label in CANDIDATES if candidates[label]["eligible"]]
    chosen = min(eligible, key=lambda label: (candidates[label]["equal_parent_mean_slice_median_total_cold_seconds"],
                                             0 if label == "corrected_batch96" else 1)) if eligible else None
    return dict(candidates=candidates, recommended_development_candidate=chosen,
        action="development recommendation only; root must separately freeze any future held-out evaluation" if chosen else "retain existing coarse96; no performance claim",
        automatic_deployment_or_heldout_evaluation=False)


def run_development(protocol_path, output, *, root=None, compute_context="unspecified"):
    root = Path(__file__).resolve().parents[3] if root is None else Path(root).resolve()
    output = Path(output).resolve()
    _require(not output.exists(), "output must be a new directory; preserve all prior evidence")
    protocol, raw, arrays, basis, inputs = load_development_inputs(protocol_path, root)
    _require(all(os.environ.get(key) == "1" for key in THREAD_KEYS), "set all five thread variables to1 before starting Python")
    provenance = environment_record(root)
    cases = _cases(arrays)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(raw)
    _json(output / "input_provenance.json", inputs)
    _json(output / "status.json", dict(status="preflight", created_utc=datetime.now(timezone.utc).isoformat(), provenance=provenance))
    rows, block_rows, receipts, vectors, first = [], [], [], [], {}
    parity_records, parity_receipts, offline_costs = [], [], {}
    rtol = protocol["parity_before_timing"]["relative_tolerance"]
    atol = protocol["parity_before_timing"]["absolute_tolerance"]
    try:
        references, offline_costs = _offline_references(cases, protocol["cohort"]["physics"], arrays, rtol, atol)
        _require(all(r["stored_teacher_agreement"] for r in references), "independent exact references differ from stored teacher labels")
        _json(output / "offline_reference_work.json", offline_costs)
        parity_records, parity_receipts = preflight(cases, basis, protocol["cohort"]["physics"], references,
                                                   rtol=rtol, atol=atol, output=output)
        _json(output / "status.json", dict(status="timing", all_parities_passed=True,
                                          recorded_utc=datetime.now(timezone.utc).isoformat()))
        load_at_timing_start = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        for repetition in range(6):
            for case in cases:
                reference = references[case["case_index"]]
                for position, method in enumerate(method_order(repetition, case["case_index"])):
                    identity = {key: case[key] for key in ("case_index", "parent_id", "slice_id")}
                    identity.update(method=method, repetition=repetition, order_position=position)
                    result = execute_control(method, config=protocol["cohort"]["physics"], basis=basis, case=case, clock=perf_counter)
                    row, blocks, direction = quality_records(result, reference, identity=identity, rtol=rtol, atol=atol)
                    key = case["case_index"], method
                    if key not in first:
                        first[key] = (result["objective"], result["gradient"])
                    prior_objective, prior_gradient = first[key]
                    row["repeat_parity"] = _parity(result["objective"], result["gradient"], prior_objective, prior_gradient, rtol, atol)
                    row["exact_reference_parity"] = None if method != "best_exact" else _parity(
                        result["objective"], result["gradient"], reference["objective"], reference["gradient"], rtol, atol)
                    row.update({name: result[name] for name in ("setup_seconds", "query_seconds", "total_cold_seconds")})
                    row["valid_timing_decomposition"] = bool(all(np.isfinite(row[name]) and row[name] >= 0
                        for name in ("setup_seconds", "query_seconds", "total_cold_seconds"))
                        and np.isclose(row["setup_seconds"] + row["query_seconds"], row["total_cold_seconds"], rtol=0, atol=1e-9))
                    fine_counts = result["query_costs"]["reference_solver"]
                    fine_zero = method == "best_exact" or ("factorizations" in fine_counts and all(value == 0 for value in fine_counts.values()))
                    if method.startswith("corrected_"):
                        fine_zero = fine_zero and all(result["query_costs"]["coarse_stability"][name] == 0 for name in (
                            "fine_grid_factorizations", "fine_grid_rhs", "fine_grid_dense_svd"))
                    row["approximation_fine_solver_work_zero"] = None if method == "best_exact" else fine_zero
                    certificate = result["certificate"]
                    receipt = {**identity, **{name: result[name] for name in ("setup_costs", "query_costs", "cumulative_costs",
                        "batch_dispatches", "exact_frequency_strategies", "exact_expected_rhs")},
                        "cost_units": COST_UNITS,
                        "stability_contexts": None if certificate is None else certificate.stability_contexts,
                        "model_sha256": None if certificate is None else certificate.model_sha256}
                    rows.append(row)
                    block_rows.extend(blocks)
                    receipts.append(receipt)
                    vectors.append((result["gradient"], direction))
                _csv(output / "cold_calls.csv", rows)
                _csv(output / "paired_cold_calls.csv", paired_rows(rows))
                _csv(output / "block_quality.csv", block_rows)
                _json(output / "cost_receipts.json", receipts)
        _, end_raw, _, _, end_inputs = load_development_inputs(protocol_path, root)
        expected = {(rep, index, method) for rep in range(6) for index in range(8) for method in METHODS}
        checks = dict(complete_fixed_cohort=len(rows) == 288 and len(block_rows) == 6912
            and {(r["repetition"], r["case_index"], r["method"]) for r in rows} == expected
            and {(r["repetition"], r["case_index"], r["method"], r["block_index"]) for r in block_rows}
                == {(*key, block) for key in expected for block in range(24)},
            parity_before_timing=len(parity_records) == 24 and all(r["passes"] for r in parity_records),
            finite_objectives_and_gradients=all(r["finite_proposal"] for r in rows),
            no_numeric_certificate_failures=not any(r["numeric_certificate_failure"] for r in rows),
            repeat_parity=all(r["repeat_parity"] for r in rows),
            exact_reference_parity=all(r["exact_reference_parity"] for r in rows if r["method"] == "best_exact"),
            valid_timing_decomposition=all(r["valid_timing_decomposition"] for r in rows),
            approximation_fine_solver_work_zero=all(r["approximation_fine_solver_work_zero"] for r in rows if r["method"] != "best_exact"),
            available_aggregate_bounds_cover=all(r["directional_bound_covers"] for r in rows if r["certificate_available"]),
            available_block_bounds_cover=all(r["directional_bound_covers"] for r in block_rows if r["certificate_available"]),
            no_false_certifications=all(not r["certified_descent"] or r["true_descent"] for r in rows),
            source_unchanged=source_digest(root) == provenance["source_sha256"],
            inputs_unchanged=inputs == end_inputs and raw == end_raw)
        recommendation = candidate_recommendation(rows, parity_records)
        if not all(checks.values()):
            recommendation.update(recommended_development_candidate=None,
                                  action="no recommendation: study completeness/integrity/numeric checks failed")
        # Complete cohort status is distinct from candidate eligibility: an
        # unavailable finite rank64 must remain a reported negative outcome.
        summary = dict(status="complete" if all(checks.values()) else "checks-failed", schema_version=1,
            study_role=protocol["study_role"], protocol_sha256=PROTOCOL_SHA256, checks=checks,
            development_recommendation=recommendation, descriptive_methods=describe_methods(rows), provenance=provenance,
            source_sha256_at_end=source_digest(root), n_cold_calls=len(rows), n_block_quality_rows=len(block_rows),
            offline_exact_verification_costs=offline_costs, parity_work_receipts="parity_work_receipts.json",
            cost_units=COST_UNITS,
            compute_context=compute_context, load_average_at_timing_start=load_at_timing_start,
            load_average_after=list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            scope=protocol["scope"], statistical_scope=protocol["reporting"]["statistical_scope"],
            numerical_scope=protocol["reporting"]["numerical_scope"], task_difference=protocol["timing"]["different_tasks"],
            calibration_or_test_arrays_loaded=False, external_test_arrays_loaded=False)
        buffer = io.BytesIO()
        valid_indices = [i for i, vector in enumerate(vectors)
                         if vector[0] is not None and np.all(np.isfinite(vector[0]))]
        n = len(references[0]["gradient"])
        retained = np.stack([vectors[i][0] for i in valid_indices]) if valid_indices else np.empty((0, n))
        np.savez_compressed(buffer, gradients=retained, gradient_row_indices=np.asarray(valid_indices, dtype=np.int64),
            directions=np.asarray([v[1] for v in vectors]), exact_gradients=np.asarray([r["gradient"] for r in references]),
            method=np.asarray([r["method"] for r in rows]), case_index=np.asarray([r["case_index"] for r in rows]),
            repetition=np.asarray([r["repetition"] for r in rows]))
        (output / "aggregate_vectors.npz").write_bytes(buffer.getvalue())
        summary["aggregate_vectors_sha256"] = sha256_file(output / "aggregate_vectors.npz")
        summary["aggregate_vector_scope"] = {"retained_finite_gradient_rows": len(valid_indices), "full_cohort_rows": len(rows),
            "mapping": "gradients map through gradient_row_indices; all directions/metadata/cold_calls retain the full cohort"}
        _json(output / "summary.json", summary)
        _json(output / "status.json", dict(status=summary["status"], completed_cold_calls=len(rows),
                                          recorded_utc=datetime.now(timezone.utc).isoformat()))
        return summary
    except Exception as exc:
        _json(output / "status.json", dict(status="failed", exception=type(exc).__name__, message=str(exc),
            completed_cold_calls=len(rows), completed_block_rows=len(block_rows),
            offline_exact_verification_costs=offline_costs, parity_records_returned=len(parity_records),
            recorded_utc=datetime.now(timezone.utc).isoformat()))
        raise
