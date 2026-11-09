"""Explicit later-selected POD comparator; preserve the original three-FNO plan."""
from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from wavecert.certificates.directional import certify_direction
from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.experiments.gradient_development import _array_digest
from wavecert.experiments.openfwi_stress import prepare_external_models
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.statistics.finite_population import finite_population_failure_summary
from wavecert.surrogates.pod import PODGalerkinSurrogate

ADDENDUM_SHA256 = "4022fabf898c8865e9d36849c144a0cecc08e26b1910b97f9b40e09f35a2751c"
LABEL = "classical_pod"
_require, _json, _inside = confirmation._require, confirmation._json, confirmation._inside


def validate_pod_addendum(addendum_path, repository_root, *, teacher_study):
    """Audit the completed development comparison without new predictions."""
    root, teacher = Path(repository_root), Path(teacher_study)
    _require(sha256_file(addendum_path) == ADDENDUM_SHA256, "prospective POD addendum bytes changed")
    addendum = _json(addendum_path)
    plan_path = _inside(root, addendum["original_three_fno_plan_path"])
    plan = confirmation.validate_confirmation_plan(plan_path, root)
    _require(addendum["original_three_fno_plan_sha256"] == sha256_file(plan_path) and
             addendum["source_calibration"]["pod_calibration_performed"] is False and
             addendum["model_labels"] == ["legacy_forward", "legacy_jvp", "development_winner", LABEL],
             "the original FNO scope and explicit uncalibrated POD comparator must remain intact")
    study = _inside(root, addendum["pod_study_path"])
    for name, checksum in addendum["pod_study_files_sha256"].items():
        _require(sha256_file(_inside(study, name)) == checksum, f"pinned POD development artifact changed: {name}")
    summary, protocol = _json(study / "summary.json"), _json(study / "protocol.json")
    candidates, selection = _json(study / "candidates.json"), _json(study / "selection.json")
    _require(summary["status"] == _json(study / "status.json")["status"] == "complete" and
             summary["source_unchanged_during_run"] is True and summary["inputs_unchanged_during_run"] is True and
             summary["calibration_or_test_arrays_loaded"] is False and summary["external_test_arrays_loaded"] is False and
             summary["candidates"] == candidates and summary["selection"] == selection and
             summary["inputs"] == _json(study / "input_provenance.json"), "POD development comparison is incomplete or changed")
    provenance = summary["provenance"]
    _require(provenance["git_revision"] == addendum["pod_source_commit"] and
             provenance["source_sha256"] == summary["source_sha256_at_end"] == addendum["pod_source_sha256"],
             "POD source provenance differs from the reviewed development run")
    # The original source history must be retained when publishing; a local
    # cherry-pick is identified in the addendum, never relabeled as published.
    confirmation._verify_training_source(provenance, root, summary["source_sha256_at_end"])
    inputs = summary["inputs"]
    _require(inputs["opened_archive_roles"] == ["train", "validation", "validation_inverse_cases"] and
             inputs["calibration_or_test_arrays_loaded"] is False and inputs["external_test_arrays_loaded"] is False,
             "POD teacher inputs must exclude all held-out roles")
    for name, checksum in inputs["verified_file_sha256"].items():
        _require(sha256_file(_inside(teacher, name)) == checksum, f"pinned POD teacher input changed: {name}")
    identities = inputs["identities"]
    _require(len(identities["train"]["parent_ids"]) == 17 and len(identities["validation"]["parent_ids"]) == 4 and
             not set(identities["train"]["parent_ids"]) & set(identities["validation"]["parent_ids"]) and
             not set(identities["train"]["slice_ids"]) & set(identities["validation"]["slice_ids"]),
             "POD teacher parent/slice isolation changed")
    expected = [(family["name"], rank) for family in protocol["basis_families"] for rank in protocol["ranks"]]
    _require(expected == [(family, rank) for family in ("state", "state+tangent") for rank in (8, 16, 32, 64)] and
             [(row["family"], row["rank"]) for row in candidates] == expected,
             "all eight frozen POD candidates must be retained in order")
    bases = {}
    for family, info in summary["basis_families"].items():
        path = _inside(study, info["artifact"])
        _require(sha256_file(path) == info["artifact_sha256"] and info["svd_evaluations"] == 1 and info["saved_rank"] == 64,
                 "POD family artifact or one-SVD prefix design changed")
        with np.load(path, allow_pickle=False) as archive:
            basis = archive["basis"].copy()
            config = json.loads(str(archive["physics_config"]))
        _require(basis.shape == (576, 64) and np.all(np.isfinite(basis)) and
                 np.linalg.norm(basis.T @ basis - np.eye(64), ord=2) < 1e-10 and
                 _array_digest(basis) == info["basis_sha256"] and config == {
                     "shape": [24, 24], "spacing": .05, "damping_width": 3, "damping_strength": 2.},
                 "POD basis/geometry/orthogonality identity changed")
        bases[family] = basis
    validation_parents = set(identities["validation"]["parent_ids"])
    for row in candidates:
        metrics = row["validation_metrics"]
        _require(row["status"] == "complete" and _array_digest(bases[row["family"]][:, :row["rank"]]) == row["basis_prefix_sha256"] and
                 row["basis_artifact"] == summary["basis_families"][row["family"]]["artifact"] and
                 row["basis_artifact_sha256"] == summary["basis_families"][row["family"]]["artifact_sha256"],
                 "POD candidate is incomplete or its evaluated basis prefix changed")
        components = metrics["parent_components"]
        _require(set(components) == validation_parents, "POD common criterion must retain every validation parent")
        for key in ("receiver_relative_squared_l2", "jvp_relative_squared_l2", "aggregate_gradient_cosine", "aggregate_neural_true_descent"):
            mean = float(np.mean([parent[key] for parent in components.values()]))
            _require(math.isfinite(mean) and math.isclose(metrics[key], mean, rel_tol=1e-12, abs_tol=1e-12),
                     "POD criterion is not the recorded equal-parent mean")
        score = metrics["receiver_relative_squared_l2"] + metrics["jvp_relative_squared_l2"] + 1 - metrics["aggregate_gradient_cosine"]
        _require(math.isclose(metrics["direction_loss"], 1 - metrics["aggregate_gradient_cosine"], abs_tol=1e-12) and
                 math.isclose(metrics["selection_score"], score, abs_tol=1e-12), "POD common selection score changed")
    chosen = min(candidates, key=lambda row: (row["validation_metrics"]["selection_score"], expected.index((row["family"], row["rank"]))))
    required = {"family": chosen["family"], "rank": chosen["rank"], "basis_artifact": chosen["basis_artifact"],
                "basis_artifact_sha256": chosen["basis_artifact_sha256"], "basis_prefix_sha256": chosen["basis_prefix_sha256"],
                "validation_selection_score": chosen["validation_metrics"]["selection_score"]}
    _require(all(selection[key] == value == addendum["selected_pod"][key] for key, value in required.items()) and
             selection["frozen_for_heldout_evaluation"] is False and selection["changes_existing_fno_selection"] is False,
             "fixed POD winner differs from the all-eight validation-only rule")
    return addendum, plan, summary


def freeze_pod_comparator(*, addendum_path, repository_root, teacher_study, selection_dir, output_dir):
    """Immutable separate comparator receipt after all-four neural selection."""
    addendum, _, summary = validate_pod_addendum(addendum_path, repository_root, teacher_study=teacher_study)
    neural = confirmation.validate_selection_freeze(selection_dir)
    confirmation.verify_prior_stage_binding(neural["provenance"], neural["runtime_contract"], repository_root)
    output = confirmation._new_directory(output_dir)
    confirmation._write_once(output, "addendum.json", Path(addendum_path).read_bytes())
    study = _inside(repository_root, addendum["pod_study_path"])
    for name, checksum in addendum["pod_study_files_sha256"].items():
        data = _inside(study, name).read_bytes()
        _require(confirmation._sha(data) == checksum, "POD input changed while freezing")
        confirmation._write_once(output, f"development/{name}", data)
    selected = addendum["selected_pod"]
    env = environment_record(repository_root)
    frozen = {"schema_version": 1, "stage": "POD-comparator-frozen-before-fresh-confirmation", "created_utc": confirmation._utc(),
        "addendum_sha256": ADDENDUM_SHA256, "original_three_fno_plan_sha256": confirmation.FROZEN_PLAN_SHA256,
        "selection_freeze_sha256": sha256_file(Path(selection_dir) / "freeze.json"), "label": LABEL,
        "selected_pod": selected, "basis_path": f"development/{selected['basis_artifact']}",
        "all_eight_candidates": [{key: value for key, value in row.items() if key != "validation_metrics"} for row in summary["candidates"]],
        "teacher_input_file_sha256": summary["inputs"]["verified_file_sha256"],
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "provenance": env, "runtime_contract": confirmation._runtime_contract(env),
        "source_calibrated_pod_gate_evaluated": False, "fresh_predictions_performed_by_this_stage": False,
        "scope": "Separate later-selected classical comparator addendum; original three-FNO plan and neural winner rule unchanged."}
    confirmation._write_once(output, "pod_freeze.json", confirmation._json_bytes(frozen))
    return frozen


def validate_pod_freeze(pod_dir, selection_dir):
    root = Path(pod_dir)
    frozen = _json(root / "pod_freeze.json")
    confirmation.validate_selection_freeze(selection_dir)
    _require(frozen["stage"] == "POD-comparator-frozen-before-fresh-confirmation" and frozen["label"] == LABEL and
             frozen["selection_freeze_sha256"] == sha256_file(Path(selection_dir) / "freeze.json") and
             frozen["addendum_sha256"] == sha256_file(root / "addendum.json") == ADDENDUM_SHA256 and
             frozen["original_three_fno_plan_sha256"] == confirmation.FROZEN_PLAN_SHA256 and
             frozen["source_calibrated_pod_gate_evaluated"] is False and frozen["fresh_predictions_performed_by_this_stage"] is False,
             "POD freeze must be tied to this addendum and the completed neural selection")
    for name, checksum in frozen["files"].items():
        _require(sha256_file(_inside(root, name)) == checksum, "immutable POD freeze input changed")
    addendum = _json(root / "addendum.json")
    required = {"addendum.json": ADDENDUM_SHA256, **{
        f"development/{name}": checksum for name, checksum in addendum["pod_study_files_sha256"].items()}}
    _require(all(frozen["files"].get(name) == checksum for name, checksum in required.items()),
             "POD freeze omits pinned all-eight development artifacts")
    candidates = _json(root / "development/candidates.json")
    _require(frozen["selected_pod"] == addendum["selected_pod"] and
             frozen["all_eight_candidates"] == [{key: value for key, value in row.items() if key != "validation_metrics"}
                                                for row in candidates] and len(frozen["all_eight_candidates"]) == 8 and
             [(row["family"], row["rank"]) for row in frozen["all_eight_candidates"]] ==
             [(family, rank) for family in ("state", "state+tangent") for rank in (8, 16, 32, 64)] and
             sha256_file(_inside(root, frozen["basis_path"])) == frozen["selected_pod"]["basis_artifact_sha256"],
             "POD freeze must retain all-eight identities and exactly the selected basis")
    return frozen


def summarize_pod_confirmation(blocks, models, plan, audits):
    """POD-only N400 model-event inference, without inventing a calibrated gate."""
    families = {family["name"] for family in plan["families"]}
    _require(len(blocks) == 800 and len(models) == 200 and
             {r["family"] for r in blocks + models} == families and
             {r["checkpoint"] for r in blocks + models} == {LABEL}, "POD confirmation must retain both families and every selected case")
    result = {}
    analysis = plan["analysis"]
    for family in plan["families"]:
        name = family["name"]
        selected = [r for r in models if r["family"] == name]
        rows = [r for r in blocks if r["family"] == name]
        expected = set(family["confirmation_selected_indices"])
        _require(len(selected) == 100 and {r["original_index"] for r in selected} == expected and len(rows) == 400,
                 "POD confirmation cannot omit or duplicate a selected original model")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["original_index"]].append(row)
            _require(row["source_calibrated_covers"] is None and row["source_calibrated_accepts"] is None and
                     row["source_calibrated_false_acceptance"] is None, "POD has no frozen source-calibrated gate")
        _require(set(grouped) == expected and all(len(group) == 4 and
                 len({(r["source_index"], r["frequency_hz"]) for r in group}) == 4 for group in grouped.values()),
                 "POD models must retain all four distinct correlated blocks")
        events = {"aggregate_neural_non_descent": [not row["aggregate_neural_true_descent"] for row in selected],
            "model_any_deterministic_false_acceptance": [any(b["deterministic_false_acceptance"] for b in grouped[row["original_index"]]) for row in selected],
            "model_deterministic_coverage_failure": [not all(b["deterministic_covers"] for b in grouped[row["original_index"]]) for row in selected]}
        cosines = np.array([row["aggregate_gradient_cosine"] for row in selected])
        _require(np.all(np.isfinite(cosines)), "nonfinite POD aggregate cosine")
        bootstrap = {"mean": float(cosines.mean()), "interval": None, "status": "withheld: raw duplicate clusters",
                     "assumption": "Conditional independent generated draws, not verified real-site inference."}
        if audits[name]["population_unique_raw_hashes"] == 400:
            means = np.random.default_rng(analysis["bootstrap_seed"]).choice(cosines,
                size=(analysis["bootstrap_resamples"], 100), replace=True).mean(axis=1)
            tail = (1 - analysis["confidence"]) / 2
            bootstrap.update(interval=np.quantile(means, [tail, 1 - tail]).tolist(), status="conditional generated-draw bootstrap",
                             seed=analysis["bootstrap_seed"], n_resamples=analysis["bootstrap_resamples"])
        result[name] = {"analysis_population_size": 400, "sampled_original_index_slots": 100, "correlated_blocks": 400,
            "primary_mean_actual_aggregate_gradient_cosine": float(cosines.mean()),
            "primary_actual_aggregate_neural_true_descent_fraction": float(np.mean([r["aggregate_neural_true_descent"] for r in selected])),
            "conditional_model_bootstrap_aggregate_cosine": bootstrap,
            "fixed_complement_hypergeometric_bounds": {key: finite_population_failure_summary(np.asarray(value),
                population_size=400, confidence=analysis["confidence"]) for key, value in events.items()},
            "source_calibrated_gate": "not evaluated; no POD calibration scale",
            "deterministic_accepted_blocks": sum(row["deterministic_accepts"] for row in rows),
            "deterministic_false_accepted_blocks": sum(row["deterministic_false_acceptance"] for row in rows)}
    return result


def pod_survey_objective_and_gradient(surrogate, physics, m, survey, receiver_indices):
    """Differentiate the sum at one shared physical m; sum actual block VJPs."""
    gradients, values = [], []
    for block in survey:
        value, gradient, _, _ = surrogate.objective_and_gradient(m, physics.source(block.source_index),
            block.observed, receiver_indices, block.frequency_hz)
        gradients.append(gradient)
        values.append(value)
    _require(bool(gradients), "POD aggregate survey cannot be empty")
    gradient, value = np.sum(gradients, axis=0), float(sum(values))
    _require(np.all(np.isfinite(gradient)) and math.isfinite(value), "nonfinite POD aggregate objective/gradient")
    return value, gradient


def _joint_gates(selection_dir, calibration_dir, pod_dir, repository_root):
    neural, calibration = confirmation.validate_calibration_freeze(selection_dir, calibration_dir)
    pod = validate_pod_freeze(pod_dir, selection_dir)
    for receipt in (neural, calibration, pod):
        confirmation.verify_prior_stage_binding(receipt["provenance"], receipt["runtime_contract"], repository_root)
    _require(neural["runtime_contract"] == calibration["runtime_contract"] == pod["runtime_contract"],
             "joint comparator stages must share the same frozen numerical runtime")
    return neural, calibration, pod


def _run_pod_confirmation(*, plan_path, repository_root, selection_dir, calibration_dir, pod_dir, data_dir, output_dir):
    """POD outcome table; source-calibrated columns explicitly remain null."""
    import torch

    from wavecert.physics.helmholtz import SurveyBlock

    root = Path(repository_root)
    plan = confirmation.validate_confirmation_plan(plan_path, root)
    _, _, frozen = _joint_gates(selection_dir, calibration_dir, pod_dir, root)
    published = _json(_inside(root, plan["old_external"]["published_preparation_manifest_path"]))
    families, data_hashes, audits = confirmation.validate_confirmation_payloads(plan, data_dir, published_receipt=published)
    output = confirmation._new_directory(output_dir)
    torch.set_num_threads(1)
    provenance = environment_record(root)
    bindings = {str(Path(plan_path).resolve()): sha256_file(plan_path),
                str((Path(selection_dir) / "freeze.json").resolve()): sha256_file(Path(selection_dir) / "freeze.json"),
                str((Path(calibration_dir) / "calibration_freeze.json").resolve()): sha256_file(Path(calibration_dir) / "calibration_freeze.json"),
                str((Path(pod_dir) / "pod_freeze.json").resolve()): sha256_file(Path(pod_dir) / "pod_freeze.json"), **data_hashes}
    status = {"status": "running", "started_utc": confirmation._utc(), "completed_blocks": 0,
              "input_sha256": bindings, "provenance": provenance}
    status_path = output / "run_status.json"
    status_path.write_bytes(confirmation._json_bytes(status))
    blocks, models, exact_vectors, pod_vectors = [], [], [], []
    current = None
    try:
        surrogate = PODGalerkinSurrogate.load(_inside(pod_dir, frozen["basis_path"]), rank=frozen["selected_pod"]["rank"])
        geometry = SurveyGeometry.from_grid((24, 24), n_sources=6, n_receivers=20, source_depth=2, receiver_depth=2, margin=2)
        sources = (geometry.source_indices[1], geometry.source_indices[-2])
        truth, physics = (Helmholtz2D(shape=(24, 24), spacing=.05, damping_width=3, damping_strength=2) for _ in range(2))
        for family, batch, raw_hashes in families:
            for index in family["confirmation_selected_indices"]:
                current = {"family": family["name"], "original_index": index, "checkpoint": LABEL}
                m_true, m_eval = prepare_external_models(batch[index])
                truth.clear_solver_cache()
                physics.clear_solver_cache()
                reference, constants, survey = {}, {}, []
                for frequency in (3., 4.):
                    for source in sources:
                        q = truth.source(source)
                        observed = truth.restrict(truth.solve_state(m_true, q, frequency), geometry.receiver_indices)
                        value, gradient, state, _ = physics.objective_and_gradient(m_eval, q, observed, geometry.receiver_indices, frequency)
                        reference[source, frequency] = (q, observed, value, gradient, state)
                        survey.append(SurveyBlock(int(source), frequency, observed, f"{source}@{frequency}"))
                    matrix = physics.receiver_resolvent_matrix(m_eval, frequency, geometry.receiver_indices)
                    constants[frequency] = (matrix, physics.receiver_resolvent_norm(m_eval, frequency,
                        geometry.receiver_indices, receiver_resolvent_matrix=matrix))
                case_rows = []
                block_gradients, block_values = [], []
                for block in survey:
                    q, observed, value, exact_gradient, exact_state = reference[block.source_index, block.frequency_hz]
                    pod_value, pod_gradient, pod_state, _ = surrogate.objective_and_gradient(m_eval, q, observed,
                        geometry.receiver_indices, block.frequency_hz)
                    norm = float(np.linalg.norm(pod_gradient))
                    _require(math.isfinite(norm) and np.all(np.isfinite(pod_gradient)) and np.all(np.isfinite(pod_state)), "nonfinite POD prediction")
                    direction = -pod_gradient / norm if norm > 1e-14 else np.zeros_like(pod_gradient)
                    matrix, alpha = constants[block.frequency_hz]
                    kappa = physics.directional_receiver_tangent_resolvent_norm(m_eval, block.frequency_hz,
                        geometry.receiver_indices, direction, receiver_resolvent_matrix=matrix)
                    certificate = certify_direction(physics=physics, surrogate=surrogate, m=m_eval, q=q,
                        observed=observed, receiver_indices=geometry.receiver_indices, frequency_hz=block.frequency_hz,
                        direction=direction, stability_mode="directional", receiver_resolvent_matrix=matrix,
                        receiver_resolvent_norm=alpha, directional_tangent_resolvent_norm=kappa)
                    derivative = float(np.dot(exact_gradient, direction))
                    error = abs(derivative - certificate.surrogate_directional_derivative)
                    row = {**current, "model_id": f"{family['name']}:{family['member']}:{index}", "raw_model_sha256": raw_hashes[index],
                        "source_index": int(block.source_index), "frequency_hz": block.frequency_hz,
                        "exact_objective": value, "neural_objective": pod_value,
                        "forward_receiver_relative_l2": relative_l2(physics.restrict(exact_state, geometry.receiver_indices), physics.restrict(pod_state, geometry.receiver_indices)),
                        "gradient_relative_l2": relative_l2(exact_gradient, pod_gradient), "gradient_cosine": cosine_similarity(exact_gradient, pod_gradient),
                        "exact_directional_derivative": derivative, "surrogate_directional_derivative": certificate.surrogate_directional_derivative,
                        "deterministic_bound": certificate.directional_error_bound, "realized_directional_error": error,
                        "deterministic_covers": bool(error <= certificate.directional_error_bound + 1e-12),
                        "deterministic_accepts": bool(certificate.certified_descent),
                        "deterministic_false_acceptance": bool(certificate.certified_descent and derivative >= 0),
                        "source_calibrated_bound": None, "source_calibration_scale": None, "source_calibrated_covers": None,
                        "source_calibrated_accepts": None, "source_calibrated_false_acceptance": None,
                        "source_calibrated_gate_status": "not evaluated; no POD calibration scale"}
                    _require(all(np.isfinite(v) for v in row.values() if isinstance(v, (float, np.floating))), "nonfinite POD block metric")
                    case_rows.append(row)
                    block_gradients.append(pod_gradient)
                    block_values.append(pod_value)
                # This is the derivative of the actual sum of four reduced
                # objectives at identical m, not a mean of directional cosines.
                pod_value, pod_gradient = pod_survey_objective_and_gradient(surrogate, physics, m_eval, survey, geometry.receiver_indices)
                summed = np.sum(block_gradients, axis=0)
                agreement = float(np.linalg.norm(summed - pod_gradient) / max(np.linalg.norm(summed), 1e-15))
                _require(agreement < 1e-10 and math.isclose(pod_value, sum(block_values), rel_tol=1e-12, abs_tol=1e-12),
                         "POD aggregate objective/gradient differs from its block sum")
                exact = np.sum([item[3] for item in reference.values()], axis=0)
                models.append({**current, "model_id": case_rows[0]["model_id"], "raw_model_sha256": raw_hashes[index], "n_blocks": 4,
                    "aggregate_exact_objective": float(sum(item[2] for item in reference.values())), "aggregate_neural_objective": pod_value,
                    "aggregate_vs_block_sum_relative_gradient_error": agreement,
                    "secondary_mean_block_gradient_cosine": float(np.mean([row["gradient_cosine"] for row in case_rows])),
                    **confirmation.aggregate_gradient_metrics(exact, pod_gradient)})
                blocks.extend(case_rows)
                exact_vectors.append(exact)
                pod_vectors.append(pod_gradient)
                confirmation._write_rows(output / "blocks.csv", blocks)
                confirmation._write_rows(output / "models.csv", models)
                status.update(completed_blocks=len(blocks), last_completed_model=current)
                status_path.write_bytes(confirmation._json_bytes(status))
        grouped = summarize_pod_confirmation(blocks, models, plan, audits)
        _joint_gates(selection_dir, calibration_dir, pod_dir, root)
        _require(source_digest(root) == provenance["source_sha256"] and
                 all(sha256_file(path) == checksum for path, checksum in bindings.items()), "POD confirmation source/frozen input changed")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, exact_gradients=np.asarray(exact_vectors), neural_gradients=np.asarray(pod_vectors),
            family=np.asarray([row["family"] for row in models]), checkpoint=np.asarray([row["checkpoint"] for row in models]),
            original_index=np.asarray([row["original_index"] for row in models]), model_id=np.asarray([row["model_id"] for row in models]))
        confirmation._write_once(output, "aggregate_gradients.npz", buffer.getvalue())
        summary = {"schema_version": 1, "status": "complete", "completed_utc": confirmation._utc(),
            "n_blocks": len(blocks), "n_model_checkpoint_evaluations": len(models), "families": grouped,
            "raw_sampling_audits": audits, "provenance": provenance, "runtime_contract": confirmation._runtime_contract(provenance),
            "source_unchanged_during_run": True, "input_files_unchanged_during_run": True, "input_sha256": bindings,
            "aggregate_vectors_sha256": sha256_file(output / "aggregate_gradients.npz"),
            "fine_physics_validation_costs": {"truth": truth.cost_snapshot(), "reference_and_certificates": physics.cost_snapshot()},
            "pod_reduced_work": surrogate.cost_snapshot(), "pod_offline_operator_projection_cost": surrogate.offline_projection_cost,
            "source_calibrated_gate": "not evaluated; no POD calibration scale",
            "scope": "Explicit later-selected POD comparator on the original fresh N400-complement folds; no real-site, trajectory, acceleration or new-method claim."}
        confirmation._write_once(output, "summary.json", confirmation._json_bytes(summary))
        status.update(status="complete", completed_blocks=len(blocks), source_unchanged_during_run=True, input_files_unchanged_during_run=True)
        status_path.write_bytes(confirmation._json_bytes(status))
        return summary
    except Exception as error:
        status.update(status="failed", completed_blocks=len(blocks), failed_unit=current, error_type=type(error).__name__, error=str(error))
        status_path.write_bytes(confirmation._json_bytes(status))
        raise


def run_joint_pod_confirmation(*, addendum_path, plan_path, repository_root, selection_dir, calibration_dir, pod_dir, data_dir, output_dir):
    """Run unchanged three-FNO study and explicit POD comparator, keeping all."""
    _require(sha256_file(addendum_path) == ADDENDUM_SHA256, "joint evaluation addendum changed")
    confirmation.validate_confirmation_plan(plan_path, repository_root)
    _joint_gates(selection_dir, calibration_dir, pod_dir, repository_root)
    output = confirmation._new_directory(output_dir)
    status = {"status": "running", "started_utc": confirmation._utc(), "model_labels": ["legacy_forward", "legacy_jvp", "development_winner", LABEL],
              "addendum_sha256": ADDENDUM_SHA256, "plan_sha256": confirmation.FROZEN_PLAN_SHA256}
    path = output / "run_status.json"
    path.write_bytes(confirmation._json_bytes(status))
    try:
        common = dict(plan_path=plan_path, repository_root=repository_root, selection_dir=selection_dir,
                      calibration_dir=calibration_dir, data_dir=data_dir)
        neural = confirmation.run_gradient_confirmation(**common, output_dir=output / "three_fno")
        pod = _run_pod_confirmation(**common, pod_dir=pod_dir, output_dir=output / "classical_pod")
        _joint_gates(selection_dir, calibration_dir, pod_dir, repository_root)
        _require(neural["status"] == pod["status"] == "complete" and neural["n_blocks"] == 2400 and pod["n_blocks"] == 800 and
                 neural["raw_sampling_audits"] == pod["raw_sampling_audits"], "all four comparator outcomes must complete on identical native cohorts")
        summary = {**status, "status": "complete", "completed_utc": confirmation._utc(),
            "three_fno_summary_sha256": sha256_file(output / "three_fno/summary.json"),
            "pod_summary_sha256": sha256_file(output / "classical_pod/summary.json"),
            "n_blocks": 3200, "n_model_checkpoint_evaluations": 800,
            "scope": "All four model labels and both families retained; original three-FNO report unchanged, classical POD comparator explicitly added after validation selection. No held-out winner selection."}
        confirmation._write_once(output, "summary.json", confirmation._json_bytes(summary))
        status.update(status="complete", completed_utc=summary["completed_utc"])
        path.write_bytes(confirmation._json_bytes(status))
        return summary
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        path.write_bytes(confirmation._json_bytes(status))
        raise
