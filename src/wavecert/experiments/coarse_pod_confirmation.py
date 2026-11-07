"""Separately frozen fifth comparator: coarse-corrected physical POD proposal."""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.experiments import pod_confirmation as pod
from wavecert.experiments.openfwi_stress import prepare_external_models
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.statistics.finite_population import finite_population_failure_summary

ADDENDUM_SHA256 = "af6331f38a9f8eac987314dda92fc778ff9c0d0d6011fba2bebb99d5f1c8724b"
LABEL = "coarse_corrected_pod"
_require, _json, _inside = confirmation._require, confirmation._json, confirmation._inside


def validate_coarse_addendum(addendum_path, repository_root):
    _require(sha256_file(addendum_path) == ADDENDUM_SHA256, "prospective fifth-comparator addendum changed")
    addendum = _json(addendum_path)
    for name, expected in (("original_three_fno_plan", confirmation.FROZEN_PLAN_SHA256),
                           ("prior_four_model_pod_addendum", pod.ADDENDUM_SHA256)):
        entry = addendum[name]
        _require(sha256_file(_inside(repository_root, entry["path"])) == entry["sha256"] == expected,
                 "fifth comparator cannot change either preceding prospective plan")
    method = addendum["fixed_method"]
    _require(method["label"] == LABEL and method["pod_rank"] == 64 and method["coarse_stability_rank"] == 96 and
             method["method_name"] == "coarse_corrected_oriented" and
             addendum["model_labels"] == ["legacy_forward", "legacy_jvp", "development_winner", pod.LABEL, LABEL],
             "fifth comparator must retain its fixed method, ranks and all five controls")
    return addendum


def _make_components(basis_path):
    # These reviewed additive modules must be merged and frozen before this
    # function is called. No held-out arrays are needed for operator setup.
    from wavecert.certificates.coarse_adjoint import PODCoarseAdjointCertifier
    from wavecert.physics.coarse_stability import FastCoarseStability
    from wavecert.surrogates.pod import PODGalerkinSurrogate

    physics = Helmholtz2D(shape=(24, 24), spacing=.05, damping_width=3, damping_strength=2.)
    surrogate = PODGalerkinSurrogate.load(basis_path, rank=64)
    stability = FastCoarseStability(physics, rank=96)
    return physics, surrogate, stability, PODCoarseAdjointCertifier(physics, surrogate, stability)


def _verify_public_module_paths(repository_root):
    """Do not freeze tracked bytes while executing another checkout's modules."""
    root = Path(repository_root).resolve() / "src"
    for name in ("wavecert.physics.coarse_stability", "wavecert.certificates.coarse_adjoint",
                 "wavecert.physics.helmholtz", "wavecert.surrogates.pod",
                 "wavecert.experiments.coarse_pod_confirmation", "wavecert.experiments.pod_confirmation",
                 "wavecert.experiments.gradient_confirmation"):
        module = import_module(name)
        _require(Path(module.__file__).resolve() == root.joinpath(*name.split(".")).with_suffix(".py"),
                 f"public pipeline module is loaded from another checkout: {name}")


def freeze_coarse_pod_comparator(*, addendum_path, repository_root, selection_dir, calibration_dir, pod_dir, output_dir):
    """New immutable method/source/runtime/operator gate before all five outcomes."""
    addendum = validate_coarse_addendum(addendum_path, repository_root)
    _, _, previous = pod._joint_gates(selection_dir, calibration_dir, pod_dir, repository_root)
    _verify_public_module_paths(repository_root)
    _require(previous["selected_pod"]["basis_artifact_sha256"] == addendum["fixed_method"]["pod_basis_artifact_sha256"],
             "fifth method must use the existing selected training-only POD basis")
    output = confirmation._new_directory(output_dir)
    data = _inside(pod_dir, previous["basis_path"]).read_bytes()
    _require(confirmation._sha(data) == addendum["fixed_method"]["pod_basis_artifact_sha256"], "POD basis changed before fifth freeze")
    confirmation._write_once(output, "pod_basis.npz", data)
    confirmation._write_once(output, "addendum.json", Path(addendum_path).read_bytes())
    _, surrogate, stability, _ = _make_components(output / "pod_basis.npz")
    env = environment_record(repository_root)
    frozen = {"schema_version": 1, "stage": "coarse-corrected-POD-frozen-before-fresh-confirmation", "created_utc": confirmation._utc(),
        "addendum_sha256": ADDENDUM_SHA256, "label": LABEL, "fixed_method": addendum["fixed_method"],
        "basis_path": "pod_basis.npz", "basis_sha256": confirmation._sha(data), "operator_signature": stability.operator_signature,
        "selection_freeze_sha256": sha256_file(Path(selection_dir) / "freeze.json"),
        "calibration_freeze_sha256": sha256_file(Path(calibration_dir) / "calibration_freeze.json"),
        "pod_freeze_sha256": sha256_file(Path(pod_dir) / "pod_freeze.json"),
        "files": {"addendum.json": ADDENDUM_SHA256, "pod_basis.npz": confirmation._sha(data)},
        "operator_setup_cost": stability.cost_snapshot(), "pod_operator_projection_cost": surrogate.offline_projection_cost,
        "provenance": env, "runtime_contract": confirmation._runtime_contract(env),
        "source_calibrated_gate_evaluated": False, "fresh_predictions_performed_by_this_stage": False,
        "scope": "Fixed later-selected physical state/adjoint proposal and aggregate residual certificate; no AD-of-composite-objective, inversion or acceleration claim."}
    confirmation._write_once(output, "coarse_pod_freeze.json", confirmation._json_bytes(frozen))
    return frozen


def validate_coarse_pod_freeze(coarse_dir, selection_dir, calibration_dir, pod_dir, repository_root):
    pod._joint_gates(selection_dir, calibration_dir, pod_dir, repository_root)
    _verify_public_module_paths(repository_root)
    root = Path(coarse_dir)
    frozen = _json(root / "coarse_pod_freeze.json")
    addendum = validate_coarse_addendum(root / "addendum.json", repository_root)
    _require(frozen["stage"] == "coarse-corrected-POD-frozen-before-fresh-confirmation" and frozen["label"] == LABEL and
             frozen["addendum_sha256"] == ADDENDUM_SHA256 and frozen["fixed_method"] == addendum["fixed_method"] and
             frozen["basis_sha256"] == addendum["fixed_method"]["pod_basis_artifact_sha256"] and
             frozen["files"] == {"addendum.json": ADDENDUM_SHA256, "pod_basis.npz": frozen["basis_sha256"]} and
             frozen["basis_path"] == "pod_basis.npz" and frozen["source_calibrated_gate_evaluated"] is False and
             frozen["fresh_predictions_performed_by_this_stage"] is False,
             "fifth-comparator freeze is missing its fixed method/operator inputs")
    for name, checksum in frozen["files"].items():
        _require(sha256_file(_inside(root, name)) == checksum, "immutable fifth-comparator input changed")
    for path, key in ((Path(selection_dir) / "freeze.json", "selection_freeze_sha256"),
                      (Path(calibration_dir) / "calibration_freeze.json", "calibration_freeze_sha256"),
                      (Path(pod_dir) / "pod_freeze.json", "pod_freeze_sha256")):
        _require(sha256_file(path) == frozen[key], "fifth-comparator prior freeze identity changed")
    confirmation.verify_prior_stage_binding(frozen["provenance"], frozen["runtime_contract"], repository_root)
    return frozen


def coarse_result_records(result, reference, *, identity):
    """Audit physical sums, shared direction, availability and honest nulls."""
    gradient = np.asarray(result.approximate_gradient, dtype=float)
    direction = np.asarray(result.direction, dtype=float)
    exact = np.sum([item[1] for item in reference.values()], axis=0)
    _require(gradient.shape == direction.shape == exact.shape and np.all(np.isfinite(gradient)) and
             np.all(np.isfinite(direction)) and math.isfinite(result.approximate_objective), "nonfinite or absent physical fifth-method proposal")
    norm = float(np.linalg.norm(gradient))
    _require(math.isfinite(norm), "physical proposal norm overflow")
    expected = -gradient / norm if norm > 1e-14 else np.zeros_like(gradient)
    _require(np.allclose(direction, expected, rtol=1e-10, atol=1e-12), "fifth certificate must use the shared negative SUM-gradient direction")
    blocks = []
    gradients, objectives, bounds = [], [], []
    keys = set()
    for block in result.block_certificates:
        key = (block.source_index, block.frequency_hz)
        _require(key in reference and key not in keys, "fifth certificate changed or duplicated source/frequency identities")
        keys.add(key)
        objective, exact_block = reference[key]
        proposal = np.asarray(block.physical_gradient, dtype=float)
        _require(proposal.shape == gradient.shape and np.all(np.isfinite(proposal)) and
                 math.isfinite(block.approximate_objective), "nonfinite physical block proposal")
        signed = float(np.dot(proposal, direction))
        _require(math.isclose(block.approximate_directional_derivative, signed, rel_tol=1e-10, abs_tol=1e-11),
                 "block signed derivative did not use the same physical aggregate direction")
        exact_signed = float(np.dot(exact_block, direction))
        error = abs(exact_signed - signed)
        bound = block.directional_error_bound
        if block.available:
            _require(bound is not None and math.isfinite(bound) and bound >= 0, "available certificate must have a finite nonnegative bound")
            bounds.append(bound)
        else:
            _require(bound is None, "unavailable block must preserve null bound")
        block_metrics = confirmation.aggregate_gradient_metrics(exact_block, proposal)
        block_upper = signed + bound if block.available else None
        block_accepts = bool(block.available and block_upper < 0)
        row = {**identity, "source_index": int(key[0]), "frequency_hz": float(key[1]),
            "exact_objective": float(objective), "approximate_objective": float(block.approximate_objective),
            "gradient_cosine": block_metrics["aggregate_gradient_cosine"],
            "exact_gradient_norm": block_metrics["aggregate_exact_gradient_norm"],
            "physical_proposal_gradient_norm": block_metrics["aggregate_neural_gradient_norm"],
            "exact_directional_derivative": exact_signed, "approximate_directional_derivative": signed,
            "realized_directional_error": error, "deterministic_bound": bound,
            "secondary_block_certified_upper_derivative": block_upper,
            "secondary_block_accepts": block_accepts,
            "secondary_block_false_acceptance": bool(block_accepts and exact_signed >= 0),
            "certificate_available": bool(block.available), "correction_applied": bool(block.correction_applied),
            "correction_skipped": bool(block.correction_skipped), "reason": block.reason,
            "deterministic_covers": bool(error <= bound + 1e-12) if block.available else None,
            "source_calibration_scale": None, "source_calibrated_accepts": None}
        blocks.append(row)
        gradients.append(proposal)
        objectives.append(block.approximate_objective)
    _require(keys == set(reference) and len(blocks) == 4 and
             np.allclose(np.sum(gradients, axis=0), gradient, rtol=1e-10, atol=1e-11) and
             math.isclose(sum(objectives), result.approximate_objective, rel_tol=1e-12, abs_tol=1e-11),
             "fifth physical aggregate differs from its complete four-block sum")
    approximate_signed = float(np.dot(gradient, direction))
    exact_signed = float(np.dot(exact, direction))
    error = abs(exact_signed - approximate_signed)
    _require(math.isclose(result.approximate_directional_derivative, approximate_signed, rel_tol=1e-10, abs_tol=1e-11),
             "aggregate signed derivative differs from the physical proposal")
    available = all(block.available for block in result.block_certificates)
    _require(result.available is available, "aggregate availability cannot omit unavailable blocks")
    _require(result.exact_fallback_required is (not result.certified_descent),
             "uncertified physical proposals must explicitly require fallback")
    if available:
        bound = float(sum(bounds))
        upper = approximate_signed + bound
        _require(math.isclose(result.directional_error_bound, bound, rel_tol=1e-12, abs_tol=1e-11) and
                 math.isclose(result.certified_upper_derivative, upper, rel_tol=1e-12, abs_tol=1e-11) and
                 result.certified_descent is (upper < 0 and norm > 1e-14), "aggregate bound/strict descent differs from complete block sum")
    else:
        bound, upper = None, None
        _require(result.directional_error_bound is None and result.certified_upper_derivative is None and
                 result.certified_descent is False and result.exact_fallback_required is True,
                 "unavailable aggregate must abstain, require fallback and preserve null bounds")
    metrics = {key.replace("neural", "approximate"): value for key, value in confirmation.aggregate_gradient_metrics(exact, gradient).items()}
    model = {**identity, "n_blocks": 4, "aggregate_exact_objective": float(sum(item[0] for item in reference.values())),
        "aggregate_approximate_objective": float(result.approximate_objective), **metrics,
        "secondary_mean_block_gradient_cosine": float(np.mean([block["gradient_cosine"] for block in blocks])),
        "aggregate_exact_directional_derivative": exact_signed, "aggregate_approximate_directional_derivative": approximate_signed,
        "aggregate_realized_directional_error": error, "aggregate_deterministic_bound": bound,
        "aggregate_certified_upper_derivative": upper, "certificate_available": available,
        "certified_descent": bool(result.certified_descent), "exact_fallback_required": bool(result.exact_fallback_required),
        "aggregate_deterministic_covers": bool(error <= bound + 1e-12) if available else None,
        "aggregate_deterministic_false_acceptance": bool(result.certified_descent and exact_signed >= 0),
        "source_calibrated_gate": "not evaluated; no fifth-method source scale"}
    return blocks, model, exact, gradient


def validate_result_context(payload, model, signature, frequencies):
    """Bind the returned proof diagnostics to this physical model/operator."""
    values = np.asarray(model)
    _require(values.shape == (signature["n"],) and not np.iscomplexobj(values) and np.all(np.isfinite(values)),
             "certificate context requires the finite real physical model vector")
    checksum = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()
    contexts = payload["stability_contexts"]
    expected = set(map(float, frequencies))
    _require(payload["method"] == "coarse_corrected_oriented" and payload["model_sha256"] == checksum and
             len(contexts) == len(expected) and {float(frequency) for frequency in contexts} == expected and
             all(context["model_sha256"] == checksum and context["operator_signature"] == signature and
                 context["frequency_hz"] == float(frequency) for frequency, context in contexts.items()),
             "certificate model/frequency/operator context differs from the evaluated case")


def summarize_coarse_confirmation(blocks, models, plan, audits):
    """All original indices, including abstentions, remain in denominators."""
    _require(len(blocks) == 800 and len(models) == 200 and {r["checkpoint"] for r in blocks + models} == {LABEL},
             "fifth comparator must retain both families and every original model/block")
    groups = {}
    for family in plan["families"]:
        rows = [r for r in models if r["family"] == family["name"]]
        block_rows = [r for r in blocks if r["family"] == family["name"]]
        expected = set(family["confirmation_selected_indices"])
        _require(len(rows) == 100 and {r["original_index"] for r in rows} == expected and len(block_rows) == 400,
                 "fifth comparator cannot drop failed or unavailable original model cases")
        grouped = defaultdict(list)
        for row in block_rows:
            grouped[row["original_index"]].append(row)
        _require(set(grouped) == expected and all(len(group) == 4 and
                 len({(r["source_index"], r["frequency_hz"]) for r in group}) == 4 for group in grouped.values()),
                 "fifth comparator must retain all four correlated blocks per original model")
        events = {"certificate_unavailable": [not r["certificate_available"] for r in rows],
            "uncertified_or_unavailable": [not r["certified_descent"] for r in rows],
            "physical_proposal_non_descent": [not r["aggregate_approximate_true_descent"] for r in rows],
            "deterministic_false_acceptance": [r["aggregate_deterministic_false_acceptance"] for r in rows],
            "available_certificate_coverage_failure": [r["certificate_available"] and not r["aggregate_deterministic_covers"] for r in rows]}
        cosines = np.asarray([r["aggregate_gradient_cosine"] for r in rows])
        _require(np.all(np.isfinite(cosines)), "nonfinite physical aggregate cosine")
        analysis = plan["analysis"]
        bootstrap = {"mean": float(cosines.mean()), "interval": None, "status": "withheld: raw duplicate clusters",
                     "assumption": "Conditional independent generated draws, not verified real-site inference."}
        if audits[family["name"]]["population_unique_raw_hashes"] == 400:
            means = np.random.default_rng(analysis["bootstrap_seed"]).choice(cosines,
                size=(analysis["bootstrap_resamples"], 100), replace=True).mean(axis=1)
            tail = (1 - analysis["confidence"]) / 2
            bootstrap.update(interval=np.quantile(means, [tail, 1 - tail]).tolist(), status="conditional generated-draw bootstrap",
                             seed=analysis["bootstrap_seed"], n_resamples=analysis["bootstrap_resamples"])
        groups[family["name"]] = {"analysis_population_size": 400, "sampled_original_index_slots": 100, "correlated_blocks": 400,
            "primary_mean_actual_aggregate_gradient_cosine": float(cosines.mean()),
            "conditional_model_bootstrap_aggregate_cosine": bootstrap,
            "primary_actual_aggregate_physical_true_descent_fraction": float(np.mean([r["aggregate_approximate_true_descent"] for r in rows])),
            "certificate_available_models": sum(r["certificate_available"] for r in rows),
            "certified_descent_models": sum(r["certified_descent"] for r in rows),
            "exact_fallback_required_models": sum(r["exact_fallback_required"] for r in rows),
            "source_calibrated_gate": "not evaluated; no fifth-method source scale",
            "fixed_complement_hypergeometric_bounds": {key: finite_population_failure_summary(np.asarray(value),
                population_size=400, confidence=plan["analysis"]["confidence"]) for key, value in events.items()},
            "coverage_scope": "Unconditional failure event over all100 sampled indices; unavailable cases reported separately, not dropped into a conditional success fraction."}
    return groups


def _run_coarse_confirmation(*, plan_path, repository_root, selection_dir, calibration_dir, pod_dir, coarse_dir, data_dir, output_dir):
    """Exact-reference validation only; never execute online exact fallback."""
    import torch

    root = Path(repository_root)
    plan = confirmation.validate_confirmation_plan(plan_path, root)
    frozen = validate_coarse_pod_freeze(coarse_dir, selection_dir, calibration_dir, pod_dir, root)
    published = _json(_inside(root, plan["old_external"]["published_preparation_manifest_path"]))
    families, data_hashes, audits = confirmation.validate_confirmation_payloads(plan, data_dir, published_receipt=published)
    output = confirmation._new_directory(output_dir)
    torch.set_num_threads(1)
    provenance = environment_record(root)
    bindings = {str(Path(plan_path).resolve()): sha256_file(plan_path),
        str((Path(coarse_dir) / "coarse_pod_freeze.json").resolve()): sha256_file(Path(coarse_dir) / "coarse_pod_freeze.json"), **data_hashes}
    status = {"status": "running", "started_utc": confirmation._utc(), "completed_blocks": 0,
              "input_sha256": bindings, "provenance": provenance}
    status_path = output / "run_status.json"
    status_path.write_bytes(confirmation._json_bytes(status))
    blocks, models, exact_vectors, approximate_vectors, directions = [], [], [], [], []
    current = None
    try:
        physics, surrogate, stability, certifier = _make_components(_inside(coarse_dir, frozen["basis_path"]))
        _require(stability.operator_signature == frozen["operator_signature"], "fixed coarse basis/operator signature changed")
        truth = Helmholtz2D(shape=(24, 24), spacing=.05, damping_width=3, damping_strength=2.)
        geometry = SurveyGeometry.from_grid((24, 24), n_sources=6, n_receivers=20, source_depth=2, receiver_depth=2, margin=2)
        sources = (geometry.source_indices[1], geometry.source_indices[-2])
        for family, batch, raw_hashes in families:
            for index in family["confirmation_selected_indices"]:
                current = {"family": family["name"], "checkpoint": LABEL, "original_index": index,
                    "model_id": f"{family['name']}:{family['member']}:{index}", "raw_model_sha256": raw_hashes[index]}
                m_true, m_eval = prepare_external_models(batch[index])
                truth.clear_solver_cache()
                physics.clear_solver_cache()
                reference, survey = {}, []
                for frequency in (3., 4.):
                    for source in sources:
                        q = truth.source(source)
                        observed = truth.restrict(truth.solve_state(m_true, q, frequency), geometry.receiver_indices)
                        value, gradient, _, _ = physics.objective_and_gradient(m_eval, q, observed, geometry.receiver_indices, frequency)
                        reference[int(source), frequency] = (value, gradient)
                        survey.append(SurveyBlock(int(source), frequency, observed, f"{source}@{frequency}"))
                result = certifier.certify_survey(m_eval, survey, geometry.receiver_indices, method="coarse_corrected_oriented")
                payload = result.to_dict()
                # Save all returned components before quality/accounting checks;
                # a later rejection retains this raw outcome in a failed attempt.
                confirmation._write_once(output, f"certificates/{family['name']}/{index}.json", confirmation._json_bytes(payload))
                validate_result_context(payload, m_eval, frozen["operator_signature"], (3., 4.))
                rows, model, exact, approximate = coarse_result_records(result, reference, identity=current)
                fine_online = result.costs["reference_solver"]
                _require(all(fine_online.get(key, 0) == 0 for key in ("factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs", "stability_rhs", "dense_inverse_rhs")),
                         "fifth online proposal/certificate unexpectedly used a fine solver")
                model["online_cost_json"] = json.dumps(result.costs, sort_keys=True, allow_nan=False)
                blocks.extend(rows)
                models.append(model)
                exact_vectors.append(exact)
                approximate_vectors.append(approximate)
                directions.append(result.direction)
                confirmation._write_rows(output / "blocks.csv", blocks)
                confirmation._write_rows(output / "models.csv", models)
                status.update(completed_blocks=len(blocks), last_completed_model=current)
                status_path.write_bytes(confirmation._json_bytes(status))
        groups = summarize_coarse_confirmation(blocks, models, plan, audits)
        validate_coarse_pod_freeze(coarse_dir, selection_dir, calibration_dir, pod_dir, root)
        _require(source_digest(root) == provenance["source_sha256"] and all(sha256_file(path) == checksum for path, checksum in bindings.items()),
                 "fifth-comparator source/frozen inputs changed during evaluation")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, exact_gradients=np.asarray(exact_vectors), approximate_gradients=np.asarray(approximate_vectors),
            common_directions=np.asarray(directions), family=np.asarray([r["family"] for r in models]),
            original_index=np.asarray([r["original_index"] for r in models]), model_id=np.asarray([r["model_id"] for r in models]))
        confirmation._write_once(output, "aggregate_gradients.npz", buffer.getvalue())
        summary = {"schema_version": 1, "status": "complete", "completed_utc": confirmation._utc(), "n_blocks": len(blocks),
            "n_model_checkpoint_evaluations": len(models), "families": groups, "raw_sampling_audits": audits,
            "provenance": provenance, "runtime_contract": confirmation._runtime_contract(provenance),
            "source_unchanged_during_run": True, "input_files_unchanged_during_run": True, "input_sha256": bindings,
            "aggregate_vectors_sha256": sha256_file(output / "aggregate_gradients.npz"),
            "fixed_operator_signature": stability.operator_signature, "online_proposal_certificate_cost": certifier.cost_snapshot(),
            "online_pod_projection_setup_cost": surrogate.offline_projection_cost,
            "fine_exact_validation_cost": {"truth": truth.cost_snapshot(), "reference": physics.cost_snapshot()},
            "source_calibrated_gate": "not evaluated; no fifth-method source scale", "online_exact_recovery_performed": False,
            "scope": "Fixed later-selected corrected physical state/adjoint proposal; shared summed-gradient direction and aggregate residual bound. Ordinary floating-point implementation, not a formally verified enclosure. N400 fixed-complement confirmation, not AD of a composite solver, an inversion run, speedup or field-data evidence."}
        confirmation._write_once(output, "summary.json", confirmation._json_bytes(summary))
        status.update(status="complete", completed_blocks=len(blocks), source_unchanged_during_run=True, input_files_unchanged_during_run=True)
        status_path.write_bytes(confirmation._json_bytes(status))
        return summary
    except Exception as error:
        status.update(status="failed", completed_blocks=len(blocks), failed_unit=current, error_type=type(error).__name__, error=str(error))
        status_path.write_bytes(confirmation._json_bytes(status))
        raise


def run_five_model_confirmation(*, addendum_path, pod_addendum_path, plan_path, repository_root, selection_dir,
                                calibration_dir, pod_dir, coarse_dir, data_dir, output_dir):
    """Preserve all earlier model reports; complete only with all five labels."""
    validate_coarse_addendum(addendum_path, repository_root)
    confirmation.validate_confirmation_plan(plan_path, repository_root)
    validate_coarse_pod_freeze(coarse_dir, selection_dir, calibration_dir, pod_dir, repository_root)
    output = confirmation._new_directory(output_dir)
    status = {"status": "running", "started_utc": confirmation._utc(), "addendum_sha256": ADDENDUM_SHA256,
              "model_labels": ["legacy_forward", "legacy_jvp", "development_winner", pod.LABEL, LABEL]}
    path = output / "run_status.json"
    path.write_bytes(confirmation._json_bytes(status))
    try:
        common = dict(plan_path=plan_path, repository_root=repository_root, selection_dir=selection_dir,
                      calibration_dir=calibration_dir, pod_dir=pod_dir, data_dir=data_dir)
        four = pod.run_joint_pod_confirmation(**common, addendum_path=pod_addendum_path, output_dir=output / "first_four_models")
        fifth = _run_coarse_confirmation(**common, coarse_dir=coarse_dir, output_dir=output / LABEL)
        _require(four["status"] == fifth["status"] == "complete" and four["n_blocks"] == 3200 and fifth["n_blocks"] == 800,
                 "all five model labels must complete; no outcome-based omission")
        previous_audits = _json(output / "first_four_models/three_fno/summary.json")["raw_sampling_audits"]
        _require(previous_audits == fifth["raw_sampling_audits"], "fifth comparator must use exactly the same native cohorts")
        validate_coarse_pod_freeze(coarse_dir, selection_dir, calibration_dir, pod_dir, repository_root)
        summary = {**status, "status": "complete", "completed_utc": confirmation._utc(), "n_blocks": 4000,
            "n_model_checkpoint_evaluations": 1000,
            "first_four_summary_sha256": sha256_file(output / "first_four_models/summary.json"),
            "fifth_summary_sha256": sha256_file(output / LABEL / "summary.json"),
            "scope": "Explicit fifth comparator added after validation feasibility; original three-FNO plan and separate raw-POD reports preserved. All models/families/outcomes retained, no held-out winner selection or inversion/acceleration claim."}
        confirmation._write_once(output, "summary.json", confirmation._json_bytes(summary))
        status.update(status="complete", completed_utc=summary["completed_utc"])
        path.write_bytes(confirmation._json_bytes(status))
        return summary
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        path.write_bytes(confirmation._json_bytes(status))
        raise
