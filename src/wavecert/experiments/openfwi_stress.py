"""Frozen-checkpoint stress evaluation on two author-distributed velocity batches.

This regenerates synthetic reference-backend observations. It is not the
official OpenFWI seismic benchmark, field-data validation, a training study,
an inversion trajectory, or an online acceleration experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wavecert.certificates.directional import certify_direction
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.statistics.finite_population import finite_population_failure_summary
from wavecert.statistics.inference import binomial_failure_summary
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass
class FrozenStressInputs:
    protocol: dict
    manifest: dict
    families: list[tuple[dict, dict, np.ndarray]]
    checkpoint_paths: dict[str, Path]
    calibrations: dict[str, dict]
    files: dict[str, str]
    uncertainty_checks: dict[str, dict]


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("manifest and protocol paths must stay within their declared root")
    return path


def _protocol_contract(protocol: dict) -> None:
    expected_geometry = {
        "shape": [24, 24], "spacing": 0.05, "n_source_positions": 6, "n_receivers": 20,
        "source_position_indices": [1, -2], "source_depth": 2, "receiver_depth": 2,
        "margin": 2, "frequencies_hz": [3.0, 4.0],
    }
    expected_preprocessing = {
        "velocity_input_units": "m/s", "velocity_output_units": "km/s",
        "velocity_scale": 0.001, "clip_velocity": False,
        "resize": {"method": "scipy.ndimage.zoom", "order": 1},
        "smoothing": {"variable": "velocity", "sigma_grid_cells": 1.6, "mode": "reflect"},
    }
    if (protocol.get("schema_version") != 1 or protocol.get("geometry") != expected_geometry
            or protocol.get("preprocessing") != expected_preprocessing):
        raise ValueError("protocol does not match the frozen external-stress geometry/preprocessing")
    families = protocol.get("families", [])
    if ({(f["name"], f["family_code"]) for f in families}
            != {("FlatVel_A", 0), ("CurveFault_A", 1)} or len(families) != 2):
        raise ValueError("both frozen velocity families must be reported")
    if any(f["expected_n_models"] != 500 or f["selected_n_models"] != 100
           or f["expected_shape"] != [500, 70, 70] or f["expected_dtype"] != "float32"
           for f in families):
        raise ValueError("the frozen design samples 100 of 500 original models per family")
    checkpoints = protocol.get("checkpoints", [])
    if (len(checkpoints) != 2 or {c["label"] for c in checkpoints}
            != {"forward_only", "derivative_informed"}):
        raise ValueError("both existing checkpoints must be reported")
    analysis = protocol["analysis"]
    if (analysis["coverage_absolute_tolerance"] != 1e-12 or analysis["confidence"] != 0.95
            or analysis["bootstrap_resamples"] != 2000 or analysis["bootstrap_seed"] != 2026110712
            or protocol["selection"]["seed"] != 2026110711
            or protocol["evaluation_scope"]["training"] is not False
            or protocol["evaluation_scope"]["recalibration"] is not False):
        raise ValueError("analysis, model selection, and no-training/recalibration choices are frozen")


def prepare_external_models(raw_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed m/s→km/s, bilinear resize, then velocity smoothing; no clipping."""
    raw = np.asarray(raw_velocity)
    if raw.shape != (70, 70) or not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("each original model must be a positive finite (70,70) velocity array")
    velocity = raw.astype(np.float64) * 0.001
    velocity = ndi.zoom(velocity, (24 / 70, 24 / 70), order=1)
    if velocity.shape != (24, 24) or np.any(velocity <= 0) or not np.all(np.isfinite(velocity)):
        raise ValueError("fixed bilinear resize produced an invalid velocity model")
    smooth = ndi.gaussian_filter(velocity, sigma=1.6, mode="reflect")
    return (1 / velocity**2).reshape(-1), (1 / smooth**2).reshape(-1)


def validate_openfwi_stress_inputs(
    *, protocol_path: str | Path, data_dir: str | Path, repository_root: str | Path,
) -> FrozenStressInputs:
    """Validate frozen bytes and selection before loading a neural model."""
    protocol_path, data_dir, root = Path(protocol_path), Path(data_dir), Path(repository_root)
    protocol = json.loads(protocol_path.read_text())
    _protocol_contract(protocol)
    addendum_path = root / "configs" / "openfwi_external_stress_analysis_addendum.json"
    addendum = json.loads(addendum_path.read_text())
    if (addendum.get("schema_version") != 1
            or addendum["protocol_sha256"] != sha256_file(protocol_path)
            or addendum["confidence"] != protocol["analysis"]["confidence"]):
        raise ValueError("analysis addendum must match the frozen protocol bytes and confidence")
    manifest_path = data_dir / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("preparation manifest does not match the exact frozen protocol bytes")
    files = {str(protocol_path.resolve()): sha256_file(protocol_path),
             str(manifest_path.resolve()): sha256_file(manifest_path),
             str(addendum_path.resolve()): sha256_file(addendum_path)}
    prepared = manifest["families"]
    if len(prepared) != 2 or len({f["name"] for f in prepared}) != 2:
        raise ValueError("preparation manifest must contain both families exactly once")
    copied_protocol, selection_record = data_dir / "protocol.json", data_dir / "protocol_record.json"
    if sha256_file(copied_protocol) != sha256_file(protocol_path):
        raise ValueError("pre-payload frozen protocol copy differs from the evaluation protocol")
    record = json.loads(selection_record.read_text())
    if record["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("pre-payload selection record does not match the protocol")
    files.update({str(copied_protocol.resolve()): sha256_file(copied_protocol),
                  str(selection_record.resolve()): sha256_file(selection_record)})
    batches, all_ids, population_hash_groups = [], set(), defaultdict(list)
    uncertainty_checks = {}
    for family in protocol["families"]:
        matches = [f for f in prepared if f["name"] == family["name"]]
        if len(matches) != 1:
            raise ValueError("prepared family differs from frozen family selection")
        entry = matches[0]
        if entry["family_code"] != family["family_code"] or entry["member"] != family["member"]:
            raise ValueError("prepared archive member or family code changed")
        path = _contained(data_dir, entry["path"])
        checksum = sha256_file(path)
        if checksum != entry["sha256"]:
            raise ValueError("raw velocity payload SHA256 mismatch")
        if (path.stat().st_size != family["expected_uncompressed_bytes"]
                or f"{zlib.crc32(path.read_bytes()) & 0xFFFFFFFF:08x}" != family["expected_crc32"]):
            raise ValueError("raw velocity payload size/CRC differs from the pre-payload archive-member pins")
        files[str(path)] = checksum
        batch = np.load(path, mmap_mode="r", allow_pickle=False)
        if (list(batch.shape) != family["expected_shape"] or str(batch.dtype) != family["expected_dtype"]
                or list(batch.shape) != entry["shape"] or str(batch.dtype) != entry["dtype"]):
            raise ValueError("raw velocity payload shape or dtype mismatch")
        expected = sorted(np.random.default_rng(np.random.SeedSequence([
            protocol["selection"]["seed"], family["family_code"],
        ])).choice(500, size=100, replace=False).tolist())
        if (entry["selected_indices"] != expected
                or record["selected_indices_before_payload"][family["name"]] != expected
                or manifest["selected_indices_before_payload"][family["name"]] != expected):
            raise ValueError("selected indices differ from the frozen seeded sample")
        ids, hashes = entry["selected_model_ids"], entry["selected_model_sha256"]
        if len(ids) != 100 or len(hashes) != 100 or len(set(ids)) != 100:
            raise ValueError("selected original-model IDs/hashes must align with the 100 indices")
        population_hashes = [hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest() for raw in batch]
        if not np.all(np.isfinite(batch)) or np.any(batch <= 0):
            raise ValueError("original raw velocities must be finite and positive without clipping")
        for index, value in enumerate(population_hashes):
            population_hash_groups[value].append((family["name"], index))
        for index, model_id, model_hash in zip(expected, ids, hashes, strict=True):
            raw = np.asarray(batch[index])
            actual = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
            if model_hash != actual or not np.all(np.isfinite(raw)) or np.any(raw <= 0):
                raise ValueError("selected raw model hash or physical velocity values are invalid")
            if model_id != f"{family['name']}:{family['member']}:{index}" or model_id in all_ids:
                raise ValueError("selected original-model identity is inconsistent or repeated")
            all_ids.add(model_id)
        duplicate_record = entry["duplicate_hash_check"]
        if (duplicate_record["original_batch_unique_models"] != len(set(population_hashes))
                or duplicate_record["selected_unique_models"] != len(set(hashes))):
            raise ValueError("recorded duplicate counts disagree with raw payload hashes")
        uncertainty_checks[family["name"]] = {
            "original_batch_models": len(batch), "original_batch_unique_raw_models": len(set(population_hashes)),
            "selected_original_indices": len(expected), "selected_unique_raw_models": len(set(hashes)),
        }
        batches.append((family, entry, batch))
    for family, check in uncertainty_checks.items():
        cross_duplicate = any(len({name for name, _ in group}) > 1
                              and any(name == family for name, _ in group)
                              for group in population_hash_groups.values())
        check["cross_family_duplicate_raw_model_present"] = cross_duplicate
        check["intervals_eligible_under_declared_no_duplicate_design"] = (
            check["original_batch_unique_raw_models"] == check["original_batch_models"]
            and check["selected_unique_raw_models"] == check["selected_original_indices"]
            and not cross_duplicate
        )
        check["duplicate_handling"] = "Keep every selected model and block; withhold intervals if duplicate sampling units would require a different design."
    checkpoint_paths, calibrations = {}, {}
    frozen_calibrations = manifest["frozen_calibration_summaries"]
    for checkpoint in protocol["checkpoints"]:
        label = checkpoint["label"]
        path = _contained(root, checkpoint["path"])
        if sha256_file(path) != checkpoint["sha256"]:
            raise ValueError("existing checkpoint SHA256 mismatch; training/tuning is not permitted")
        files[str(path)] = checkpoint["sha256"]
        copies = [c for c in frozen_calibrations if c["label"] == label]
        if len(copies) != 1:
            raise ValueError("one frozen existing calibration summary is required per checkpoint")
        copy = copies[0]
        calibration_path = _contained(data_dir, copy["path"])
        if (copy["source_path"] != checkpoint["calibration_summary_path"]
                or copy["sha256"] != checkpoint["calibration_summary_sha256"]
                or sha256_file(calibration_path) != copy["sha256"]):
            raise ValueError("frozen source-domain calibration summary SHA256 mismatch")
        calibration = json.loads(calibration_path.read_text())
        scale = calibration["conformal_scale"]
        if (calibration["checkpoint_sha256"] != checkpoint["sha256"]
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError("source-domain scale is invalid or linked to a different checkpoint")
        files[str(calibration_path)] = copy["sha256"]
        checkpoint_paths[label], calibrations[label] = path, calibration
    return FrozenStressInputs(protocol, manifest, batches, checkpoint_paths, calibrations, files, uncertainty_checks)


def _model_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["checkpoint"], row["model_id"])].append(row)
    result = []
    for (family, checkpoint, model_id), group in sorted(grouped.items()):
        if len(group) != 4 or len({(r["source_index"], r["frequency_hz"]) for r in group}) != 4:
            raise ValueError("each original model/checkpoint must retain all four distinct blocks")
        result.append({
            "family": family, "checkpoint": checkpoint, "model_id": model_id,
            "original_index": group[0]["original_index"], "n_blocks": 4,
            "raw_model_sha256": group[0]["raw_model_sha256"],
            "forward_receiver_relative_l2_median": float(np.median([r["forward_receiver_relative_l2"] for r in group])),
            "gradient_cosine_mean": float(np.mean([r["gradient_cosine"] for r in group])),
            "gradient_cosine_median": float(np.median([r["gradient_cosine"] for r in group])),
            "neural_true_descent_fraction": float(np.mean([r["neural_true_descent"] for r in group])),
            "all_blocks_deterministic_covered": all(r["deterministic_covers"] for r in group),
            "all_blocks_source_calibrated_covered": all(r["source_calibrated_covers"] for r in group),
            "any_deterministic_false_acceptance": any(r["deterministic_false_acceptance"] for r in group),
            "any_source_calibrated_false_acceptance": any(r["source_calibrated_false_acceptance"] for r in group),
        })
    return result


def summarize_stress_rows(
    rows: list[dict], protocol: dict, uncertainty_checks: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Keep correlated blocks together and never pool the two environments."""
    models = _model_rows(rows)
    result = {}
    analysis = protocol["analysis"]
    for family in protocol["families"]:
        result[family["name"]] = {}
        for checkpoint in protocol["checkpoints"]:
            selected = [m for m in models if m["family"] == family["name"] and m["checkpoint"] == checkpoint["label"]]
            blocks = [r for r in rows if r["family"] == family["name"] and r["checkpoint"] == checkpoint["label"]]
            if len(selected) != family["selected_n_models"] or len(blocks) != 4 * len(selected):
                raise ValueError("family/checkpoint evaluation is incomplete")
            events = {
                "model_any_source_calibrated_false_acceptance": [m["any_source_calibrated_false_acceptance"] for m in selected],
                "model_source_calibrated_coverage_failure": [not m["all_blocks_source_calibrated_covered"] for m in selected],
                "model_any_deterministic_false_acceptance": [m["any_deterministic_false_acceptance"] for m in selected],
                "model_deterministic_coverage_failure": [not m["all_blocks_deterministic_covered"] for m in selected],
            }
            uncertainty = {}
            for name, failures in events.items():
                if not uncertainty_checks[family["name"]]["intervals_eligible_under_declared_no_duplicate_design"]:
                    uncertainty[name] = {"status": "unavailable: duplicate-unit sampling design",
                        "reason": "Raw duplicate clusters retained without resampling/discarding. Collapsed clusters are not a simple uniform 100-of-500 sample or established independent generated draws.",
                        "observed_model_index_failures": int(np.sum(failures)), "n_selected_indices": len(failures)}
                    continue
                conditional = binomial_failure_summary(np.asarray(failures), confidence=analysis["confidence"])
                conditional["assumption"] = "Conditional independent generated-model Bernoulli draw assumption; not verified IID or real-site inference."
                uncertainty[name] = {
                    "fixed_batch_uniform_sample": finite_population_failure_summary(
                        np.asarray(failures), population_size=family["expected_n_models"],
                        confidence=analysis["confidence"],
                    ), "conditional_generated_draw_binomial": conditional,
                }
            values = np.asarray([m["gradient_cosine_mean"] for m in selected])
            bootstrap = {"status": "unavailable: duplicate-unit sampling design", "mean": float(np.mean(values)), "interval": None}
            if uncertainty_checks[family["name"]]["intervals_eligible_under_declared_no_duplicate_design"]:
                rng = np.random.default_rng(analysis["bootstrap_seed"])
                means = np.mean(rng.choice(values, size=(analysis["bootstrap_resamples"], len(values)), replace=True), axis=1)
                tail = (1-analysis["confidence"]) / 2
                bootstrap = {"status": "conditional generated-draw interval",
                    "mean": float(np.mean(values)), "interval": [float(v) for v in np.quantile(means, [tail, 1-tail])],
                    "confidence": analysis["confidence"], "n_resamples": analysis["bootstrap_resamples"],
                    "seed": analysis["bootstrap_seed"], "estimand": "mean of per-original-model mean gradient cosines",
                    "assumption": "Conditional on independently generated model draws; not independently sampled real sites or another batch."}
            result[family["name"]][checkpoint["label"]] = {
                "n_original_generated_models": len(selected), "n_correlated_source_frequency_blocks": len(blocks),
                "forward_receiver_relative_l2_median": float(np.median([r["forward_receiver_relative_l2"] for r in blocks])),
                "gradient_cosine_median": float(np.median([r["gradient_cosine"] for r in blocks])),
                "neural_true_descent_block_fraction": float(np.mean([r["neural_true_descent"] for r in blocks])),
                "deterministic_block_coverage": float(np.mean([r["deterministic_covers"] for r in blocks])),
                "source_calibrated_block_coverage": float(np.mean([r["source_calibrated_covers"] for r in blocks])),
                "model_all_blocks_deterministic_coverage": float(np.mean([m["all_blocks_deterministic_covered"] for m in selected])),
                "model_all_blocks_source_calibrated_coverage": float(np.mean([m["all_blocks_source_calibrated_covered"] for m in selected])),
                "deterministic_accepted_blocks": sum(r["deterministic_accepts"] for r in blocks),
                "source_calibrated_accepted_blocks": sum(r["source_calibrated_accepts"] for r in blocks),
                "deterministic_false_accepted_blocks": sum(r["deterministic_false_acceptance"] for r in blocks),
                "source_calibrated_false_accepted_blocks": sum(r["source_calibrated_false_acceptance"] for r in blocks),
                "model_event_uncertainty": uncertainty,
                "raw_duplicate_sampling_audit": uncertainty_checks[family["name"]],
                "conditional_model_bootstrap_mean_gradient_cosine": bootstrap,
            }
    return models, result


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_openfwi_stress(
    *, protocol_path: str | Path, data_dir: str | Path, output_dir: str | Path,
    repository_root: str | Path,
) -> dict:
    """Execute the frozen design; callers must not tune from these outcomes."""
    root, output = Path(repository_root), Path(output_dir)
    inputs = validate_openfwi_stress_inputs(protocol_path=protocol_path, data_dir=data_dir, repository_root=root)
    protocol = inputs.protocol
    import torch
    torch.set_num_threads(1)
    provenance = environment_record(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(Path(protocol_path).read_bytes())
    (output / "analysis_addendum.json").write_bytes((root / "configs" / "openfwi_external_stress_analysis_addendum.json").read_bytes())
    (output / "preparation_manifest.json").write_bytes((Path(data_dir) / "preparation_manifest.json").read_bytes())
    start = perf_counter()
    rows = []
    current = None
    status = {"status": "running", "completed_blocks": 0, "provenance": provenance,
              "input_sha256": inputs.files}
    status_path = output / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    try:
        surrogates = {label: TrainedFNOWavefieldSurrogate.load(path, device="cpu")
                      for label, path in inputs.checkpoint_paths.items()}
        if any(s.shape != (24, 24) for s in surrogates.values()):
            raise ValueError("both frozen checkpoints must use the declared 24x24 grid")
        geometry = SurveyGeometry.from_grid((24, 24), n_sources=6, n_receivers=20,
                                            source_depth=2, receiver_depth=2, margin=2)
        sources = (int(geometry.source_indices[1]), int(geometry.source_indices[-2]))
        truth = Helmholtz2D(shape=(24, 24), spacing=0.05, damping_width=3, damping_strength=2.0)
        physics = Helmholtz2D(shape=(24, 24), spacing=0.05, damping_width=3, damping_strength=2.0)
        for family, entry, batch in inputs.families:
            for position, index in enumerate(entry["selected_indices"]):
                current = {"family": family["name"], "original_index": index}
                m_true, m_eval = prepare_external_models(batch[index])
                physics.clear_solver_cache()
                truth.clear_solver_cache()
                reference, constants = {}, {}
                for frequency in (3.0, 4.0):
                    for source in sources:
                        q = truth.source(source)
                        observed = truth.restrict(truth.solve_state(m_true, q, frequency), geometry.receiver_indices)
                        exact_objective, gradient, state, _ = physics.objective_and_gradient(
                            m_eval, q, observed, geometry.receiver_indices, frequency,
                        )
                        reference[source, frequency] = (q, observed, gradient, state, exact_objective)
                    matrix = physics.receiver_resolvent_matrix(m_eval, frequency, geometry.receiver_indices)
                    alpha = physics.receiver_resolvent_norm(m_eval, frequency, geometry.receiver_indices,
                                                           receiver_resolvent_matrix=matrix)
                    constants[frequency] = (matrix, alpha)
                for label, surrogate in surrogates.items():
                    scale = float(inputs.calibrations[label]["conformal_scale"])
                    for frequency in (3.0, 4.0):
                        for source in sources:
                            current = {**current, "checkpoint": label, "source_index": source, "frequency_hz": frequency}
                            q, observed, exact_gradient, exact_state, exact_objective = reference[source, frequency]
                            neural_objective, neural_gradient, neural_state, _ = surrogate.objective_and_gradient(
                                m_eval, q, observed, geometry.receiver_indices, frequency,
                            )
                            if (np.asarray(neural_gradient).shape != (physics.n,)
                                    or not np.all(np.isfinite(neural_gradient)) or not np.all(np.isfinite(neural_state))):
                                raise ValueError("non-finite or incorrectly shaped neural result")
                            norm = float(np.linalg.norm(neural_gradient))
                            direction = np.zeros(physics.n) if norm <= 1e-14 else -neural_gradient / norm
                            matrix, alpha = constants[frequency]
                            kappa = physics.directional_receiver_tangent_resolvent_norm(
                                m_eval, frequency, geometry.receiver_indices, direction,
                                receiver_resolvent_matrix=matrix,
                            )
                            certificate = certify_direction(
                                physics=physics, surrogate=surrogate, m=m_eval, q=q, observed=observed,
                                receiver_indices=geometry.receiver_indices, frequency_hz=frequency,
                                direction=direction, stability_mode="directional",
                                receiver_resolvent_matrix=matrix, receiver_resolvent_norm=alpha,
                                directional_tangent_resolvent_norm=kappa, validate_exact=False,
                            )
                            exact_dd = float(np.dot(exact_gradient, direction))
                            neural_dd = certificate.surrogate_directional_derivative
                            bound = certificate.directional_error_bound
                            scaled = scale * bound
                            realized = abs(exact_dd - neural_dd)
                            tolerance = protocol["analysis"]["coverage_absolute_tolerance"]
                            deterministic_accepts = neural_dd + bound < 0
                            scaled_accepts = neural_dd + scaled < 0
                            row = {
                                "family": family["name"], "checkpoint": label,
                                "archive_member": family["member"], "original_index": index,
                                "model_id": entry["selected_model_ids"][position],
                                "raw_model_sha256": entry["selected_model_sha256"][position],
                                "source_index": int(source), "frequency_hz": frequency,
                                "forward_receiver_relative_l2": relative_l2(
                                    physics.restrict(exact_state, geometry.receiver_indices),
                                    physics.restrict(neural_state, geometry.receiver_indices)),
                                "gradient_relative_l2": relative_l2(exact_gradient, neural_gradient),
                                "gradient_cosine": cosine_similarity(exact_gradient, neural_gradient),
                                "reference_objective": float(exact_objective),
                                "neural_objective": float(neural_objective),
                                "exact_gradient_norm": float(np.linalg.norm(exact_gradient)),
                                "neural_gradient_norm": norm,
                                "exact_directional_derivative": exact_dd,
                                "surrogate_directional_derivative": neural_dd,
                                "realized_directional_error": realized, "deterministic_bound": bound,
                                "source_calibration_scale": scale, "source_calibrated_bound": scaled,
                                "deterministic_covers": bool(realized <= bound + tolerance),
                                "source_calibrated_covers": bool(realized <= scaled + tolerance),
                                "neural_true_descent": bool(exact_dd < 0),
                                "deterministic_accepts": bool(deterministic_accepts),
                                "source_calibrated_accepts": bool(scaled_accepts),
                                "deterministic_false_acceptance": bool(deterministic_accepts and exact_dd >= 0),
                                "source_calibrated_false_acceptance": bool(scaled_accepts and exact_dd >= 0),
                            }
                            if any(not np.isfinite(value) for value in row.values()
                                   if isinstance(value, (int, float, np.integer, np.floating))):
                                raise ValueError("non-finite output metric")
                            rows.append(row)
                _write_csv(output / "blocks.csv", rows)
                status.update(completed_blocks=len(rows), last_completed_model=current)
                status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        models, grouped_summary = summarize_stress_rows(rows, protocol, inputs.uncertainty_checks)
        _write_csv(output / "models.csv", models)
        end_source = source_digest(root)
        input_unchanged = all(sha256_file(path) == checksum for path, checksum in inputs.files.items())
        unchanged = end_source == provenance["source_sha256"] and input_unchanged
        summary = {
            "schema_version": 1, "study_status": "complete" if unchanged else "stale-input-or-source-changed",
            "source_unchanged_during_run": end_source == provenance["source_sha256"],
            "input_files_unchanged_during_run": input_unchanged, "source_sha256_at_end": end_source,
            "protocol_sha256": sha256_file(protocol_path), "input_sha256": inputs.files,
            "analysis_addendum": json.loads((output / "analysis_addendum.json").read_text()),
            "n_blocks": len(rows), "n_model_checkpoint_evaluations": len(models),
            "families": grouped_summary, "provenance": provenance,
            "elapsed_seconds": perf_counter() - start,
            "reference_physics": {"backend": "Helmholtz2D", "damping_width": 3, "damping_strength": 2.0},
            "offline_evaluation_costs": {"truth_generation": truth.cost_snapshot(),
                "exact_reference_and_certificate_evaluation": physics.cost_snapshot(),
                "scope": "Offline validation work, including truth synthesis, shared exact gradients, stability maps and derivative checks. Not an online inversion cost or acceleration measurement."},
            "source_domain_calibration": {label: {"conformal_scale": c["conformal_scale"],
                "checkpoint_sha256": c["checkpoint_sha256"]} for label, c in inputs.calibrations.items()},
            "interpretation_limits": [
                "New assistant-development external stress study, not student preregistration or an official OpenFWI seismic benchmark.",
                "Synthetic observations regenerated after resizing two author-provided generated velocity batches; no field data or original seismic records.",
                "Both frozen checkpoints and every selected model/block are reported without training, tuning, recalibration, or model selection.",
                "Source-calibrated OOD coverage and acceptance are descriptive; nominal conformal coverage and exchangeability do not apply.",
                "The original generated model is the statistical unit; four correlated blocks are never four independent models.",
                "Fixed-batch hypergeometric bounds cover only each named 500-model batch under uniform sampling without replacement; no inference to other batches, sites, or families.",
                "Binomial/bootstrap intervals additionally require an independent generated-model draw assumption that is not verified real-site independence.",
                "Families and checkpoints remain separate; intervals are not simultaneous across reported groups/outcomes.",
                "Ordinary numerical discrete bounds do not formally enclose roundoff; direction signs are not finite-step, adaptive-path, or trajectory guarantees.",
                "Angles and strict directional signs can be ill-conditioned for near-stationary models; raw exact/neural objectives and gradient norms are retained, with no post-hoc filtering or sign-threshold changes.",
            ],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True, sharey=True)
        colors = {"FlatVel_A": "#0072B2", "CurveFault_A": "#D55E00"}
        for axis, checkpoint in zip(axes, protocol["checkpoints"], strict=True):
            for family in protocol["families"]:
                selected = [m for m in models if m["family"] == family["name"] and m["checkpoint"] == checkpoint["label"]]
                axis.scatter([m["forward_receiver_relative_l2_median"] for m in selected],
                    [m["gradient_cosine_median"] for m in selected], color=colors[family["name"]],
                    label=family["name"], s=17, alpha=0.65)
            axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
            axis.set_title(checkpoint["label"].replace("_", " "))
            axis.set_xlabel("Per-model median receiver relative L2 error")
        axes[0].set_ylabel("Per-model median gradient cosine")
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle("External generated-velocity stress test\nOne point per original model; four source/frequency blocks")
        fig.savefig(output / "model_forward_error_vs_gradient_cosine.png", dpi=180)
        plt.close(fig)
        status.update(status=summary["study_status"], completed_blocks=len(rows),
                      source_unchanged_during_run=summary["source_unchanged_during_run"],
                      input_files_unchanged_during_run=input_unchanged)
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        if not unchanged:
            raise RuntimeError("frozen source or input bytes changed during external evaluation")
        return summary
    except Exception as error:
        if rows:
            _write_csv(output / "blocks.csv", rows)
        status.update(status="failed" if not str(status["status"]).startswith("stale") else status["status"],
                      completed_blocks=len(rows), failed_unit=current,
                      error_type=type(error).__name__, error=str(error))
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        raise
