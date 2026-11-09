"""Explicit historical-test reuse: fixed models, complete parent groups, no IID claims."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from wavecert.data.splits import audit_split_manifest
from wavecert.experiments import coarse_pod_confirmation as coarse
from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.experiments import pod_confirmation as pod
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate
from wavecert.surrogates.pod import PODGalerkinSurrogate

PROTOCOL_SHA256 = "b5ecc87c5d90e19a39040e4f57a2d88ac9da0d1b54991e574fdb0839ffaf2f1d"
LABELS = ("legacy_forward", "legacy_jvp", "development_winner", "classical_pod", "coarse_corrected_pod")
_require, _json, _inside = confirmation._require, confirmation._json, confirmation._inside
PHASE_FILES = {"configs/historical_transfer_protocol.json", "scripts/run_historical_transfer.py",
               "src/wavecert/experiments/historical_transfer.py"}


def _csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit_historical_metadata(protocol, dataset_dir):
    """Inspect only manifests/metadata. Never open a test NPZ here."""
    root = Path(dataset_dir)
    cfg = protocol["dataset"]
    _require(sha256_file(root / "dataset_manifest.json") == cfg["dataset_manifest_sha256"] and
             sha256_file(root / "split_manifest.csv") == cfg["role_manifest_sha256"], "historical dataset/parent role manifest changed")
    manifest = _json(root / "dataset_manifest.json")
    audit = audit_split_manifest(_csv(root / "split_manifest.csv"))
    _require(audit["parent_isolation_passed"] and audit["slice_isolation_passed"], "historical parent/slice roles overlap")
    sources, frequencies = protocol["geometry"]["source_indices"], protocol["geometry"]["frequencies_hz"]
    pairs = {(source, frequency) for source in sources for frequency in frequencies}
    _require(len(pairs) == protocol["geometry"]["blocks_per_slice"] == 24, "historical design must retain every original24 block")
    metadata = {}
    for role, expected in cfg["roles"].items():
        _require(role in {"test_id", "test_ood"} and sha256_file(root / expected["metadata_file"]) == expected["metadata_sha256"] and
                 manifest["splits"][role]["arrays_sha256"] == expected["arrays_sha256"] and
                 manifest["splits"][role]["metadata_sha256"] == expected["metadata_sha256"], "historical cached role identity changed")
        rows = _csv(root / expected["metadata_file"])
        grouped = defaultdict(list)
        parents = defaultdict(set)
        for row in rows:
            _require(row["split"] == role, "historical metadata role changed")
            grouped[row["slice_id"]].append(row)
            parents[row["model_id"]].add(row["slice_id"])
        observed = {parent: sorted(slices) for parent, slices in sorted(parents.items())}
        _require(observed == expected["parents"] and len(rows) == expected["n_blocks"] and
                 len(grouped) == expected["n_slices"] and len(parents) == expected["n_parents"] and
                 set(parents) == set(audit["roles"][role]["parent_models"]), "historical cohort must retain every pinned parent and slice")
        for slice_id, group in grouped.items():
            _require(len(group) == 24 and {(int(row["source_index"]), float(row["frequency_hz"])) for row in group} == pairs and
                     len({row["model_id"] for row in group}) == 1 and len({row["source_sha256"] for row in group}) == 1,
                     f"historical slice requires all24 distinct shared-model blocks: {slice_id}")
        metadata[role] = rows
    _require(cfg["roles"]["test_id"]["n_parents"] == 8 and cfg["roles"]["test_id"]["n_slices"] == 16 and
             cfg["roles"]["test_ood"]["n_parents"] == 1 and cfg["roles"]["test_ood"]["n_slices"] == 4,
             "historical cohort is fixed at8 ID parents/16 slices and1 OOD parent/4 slices")
    return metadata, audit


def validate_historical_protocol(protocol_path, repository_root, *, dataset_dir=None):
    _require(sha256_file(protocol_path) == PROTOCOL_SHA256, "prospective historical-transfer protocol changed")
    protocol = _json(protocol_path)
    _require(protocol["model_labels"] == list(LABELS) and set(protocol["staging"]["historical_new_source_allowlist"]) == PHASE_FILES and
             protocol["method_parameters"]["no_learning_selection_recalibration"] is True, "historical phase scope/allowlist changed")
    for name, checksum in protocol["required_prior_plans"].items():
        _require(sha256_file(_inside(repository_root, f"configs/{name}")) == checksum, "historical phase cannot change earlier prospective plans")
    dataset = Path(dataset_dir) if dataset_dir is not None else _inside(repository_root, protocol["dataset"]["path"])
    metadata, audit = audit_historical_metadata(protocol, dataset)
    return protocol, metadata, audit


def _clean_prior_audit(repository_root, selection_dir, calibration_dir, pod_dir, coarse_dir):
    """Run unchanged old guards in a fresh process with no new phase imports."""
    code = '''import json,sys
from pathlib import Path
import torch
torch.set_num_threads(1)
from wavecert.experiments.coarse_pod_confirmation import validate_coarse_pod_freeze
from wavecert.provenance import sha256_file
root,B,C,P,E=map(Path,sys.argv[1:])
f=validate_coarse_pod_freeze(E,B,C,P,root)
print(json.dumps({"status":"prior guards passed in clean process", "old_fifth_provenance":f["provenance"],"runtime_contract":f["runtime_contract"],"prior_receipt_sha256":{name:sha256_file(path) for name,path in {"B":B/"freeze.json","C":C/"calibration_freeze.json","POD":P/"pod_freeze.json","fifth":E/"coarse_pod_freeze.json"}.items()}},allow_nan=False))
'''
    root = Path(repository_root).resolve()
    args = [str(Path(path).resolve()) for path in (root, selection_dir, calibration_dir, pod_dir, coarse_dir)]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", code, *args], cwd=root, env=env, capture_output=True, text=True)
    _require(result.returncode == 0, f"unchanged prior freeze guards rejected historical admission: {result.stderr[-2500:]}")
    receipt = json.loads(result.stdout)
    receipt["audit_code_sha256"] = confirmation._sha(code.encode())
    receipt["scope"] = "No historical module/CLI imported; prior source/runtime/receipt/weight guards unchanged. No model prediction or historical test-array loading."
    return receipt


def phase_source_admission(previous_provenance, current, repository_root):
    """Admit only explicitly allowlisted imported additions, preserving old bytes."""
    root = Path(repository_root).resolve()
    old, new = previous_provenance["source_files"], current["source_files"]
    _require(all(new.get(name) == checksum for name, checksum in old.items()), "previously frozen source changed before historical phase")
    _require(PHASE_FILES <= set(new) and Path(__file__).resolve() == _inside(root, "src/wavecert/experiments/historical_transfer.py"),
             "historical phase files must execute from the frozen checkout")
    additions = set(new) - set(old)
    imported = {Path(module.__file__).resolve() for module in tuple(sys.modules.values()) if getattr(module, "__file__", None)}
    _require(not any(_inside(root, name) in imported for name in additions - PHASE_FILES),
             "unapproved new imported source in historical phase")
    return {"old_recorded_source_files_unchanged": True, "explicit_phase_source_files": {name: new[name] for name in sorted(PHASE_FILES)},
        "added_phase_files": sorted(additions & PHASE_FILES), "added_unimported_unrelated_files": sorted(additions - PHASE_FILES),
        "scope": "New prospective historical phase only; all earlier frozen source paths remain unchanged, no old guard weakened."}


def _verify_historical_module_paths(repository_root):
    coarse._verify_public_module_paths(repository_root)
    root = Path(repository_root).resolve()
    fno_module = sys.modules[TrainedFNOWavefieldSurrogate.__module__]
    _require(Path(fno_module.__file__).resolve() == _inside(root, "src/wavecert/surrogates/fno.py") and
             Path(__file__).resolve() == _inside(root, "src/wavecert/experiments/historical_transfer.py"),
             "historical/FNO module executes from another checkout")


def freeze_historical_transfer(*, protocol_path, repository_root, selection_dir, calibration_dir, pod_dir, coarse_dir, output_dir, dataset_dir=None):
    """Snapshot old weights/receipts and a new complete source/runtime contract."""
    import torch

    protocol, _, audit = validate_historical_protocol(protocol_path, repository_root, dataset_dir=dataset_dir)
    prior_audit = _clean_prior_audit(repository_root, selection_dir, calibration_dir, pod_dir, coarse_dir)
    _verify_historical_module_paths(repository_root)
    torch.set_num_threads(1)
    current = environment_record(repository_root)
    _require(confirmation._runtime_contract(current) == prior_audit["runtime_contract"], "historical numerical runtime differs from frozen models")
    admission = phase_source_admission(prior_audit["old_fifth_provenance"], current, repository_root)
    output = confirmation._new_directory(output_dir)
    confirmation._write_once(output, "protocol.json", Path(protocol_path).read_bytes())
    confirmation._write_once(output, "prior_guard_audit.json", confirmation._json_bytes(prior_audit))
    confirmation._write_once(output, "source_admission.json", confirmation._json_bytes(admission))
    prior = {}
    for name, directory, filename in (("B", selection_dir, "freeze.json"), ("C", calibration_dir, "calibration_freeze.json"),
                                       ("POD", pod_dir, "pod_freeze.json"), ("fifth", coarse_dir, "coarse_pod_freeze.json")):
        data = (Path(directory) / filename).read_bytes()
        _require(confirmation._sha(data) == prior_audit["prior_receipt_sha256"][name], "prior receipt changed during historical freeze")
        confirmation._write_once(output, f"prior/{name}.json", data)
        prior[name] = json.loads(data)
    models = []
    for label in LABELS[:3]:
        model = next(item for item in prior["B"]["models"] if item["label"] == label)
        data = _inside(selection_dir, model["checkpoint_path"]).read_bytes()
        _require(confirmation._sha(data) == model["checkpoint_sha256"], "FNO weights changed during historical freeze")
        path = f"models/{model['label']}.pt"
        confirmation._write_once(output, path, data)
        models.append({"label": model["label"], "kind": "fno", "path": path, "sha256": model["checkpoint_sha256"]})
    basis = _inside(coarse_dir, prior["fifth"]["basis_path"]).read_bytes()
    _require(confirmation._sha(basis) == protocol["method_parameters"]["pod_basis_artifact_sha256"] == prior["fifth"]["basis_sha256"], "historical POD basis identity changed")
    confirmation._write_once(output, "models/pod_basis.npz", basis)
    models.extend({"label": label, "kind": kind, "path": "models/pod_basis.npz", "sha256": confirmation._sha(basis)}
                  for label, kind in ((pod.LABEL, "pod"), (coarse.LABEL, "physical_coarse_adjoint")))
    _require(source_digest(repository_root) == current["source_sha256"], "source changed during historical freeze")
    frozen = {"schema_version": 1, "stage": "historical-transfer-frozen-before-new-project-outcomes", "created_utc": confirmation._utc(),
        "protocol_sha256": PROTOCOL_SHA256, "models": models, "operator_signature": prior["fifth"]["operator_signature"],
        "prior_receipt_sha256": prior_audit["prior_receipt_sha256"], "source_admission": admission,
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "provenance": current, "runtime_contract": confirmation._runtime_contract(current), "parent_role_audit": audit,
        "project_test_arrays_loaded_by_this_stage": False, "new_model_outcomes_computed_by_this_stage": False,
        "development_training_admission": prior["B"].get("training_admission"),
        "scope": "Explicit historical test reuse/development-transfer freeze; prior source/runtime/weight guards preserved, no learning/reselection/calibration or pristine-confirmation claim."}
    confirmation._write_once(output, "historical_freeze.json", confirmation._json_bytes(frozen))
    return frozen


def validate_historical_freeze(freeze_dir, repository_root):
    _verify_historical_module_paths(repository_root)
    root = Path(freeze_dir)
    frozen = _json(root / "historical_freeze.json")
    _require(frozen["stage"] == "historical-transfer-frozen-before-new-project-outcomes" and
             frozen["protocol_sha256"] == sha256_file(root / "protocol.json") == PROTOCOL_SHA256 and
             [model["label"] for model in frozen["models"]] == list(LABELS) and
             frozen["project_test_arrays_loaded_by_this_stage"] is False and frozen["new_model_outcomes_computed_by_this_stage"] is False,
             "complete historical phase freeze is required before project outcomes")
    required = {"protocol.json", "prior_guard_audit.json", "source_admission.json", "prior/B.json", "prior/C.json", "prior/POD.json", "prior/fifth.json"}
    required |= {model["path"] for model in frozen["models"]}
    _require(required <= set(frozen["files"]), "historical freeze lacks prior receipts and model copies")
    for name, checksum in frozen["files"].items():
        _require(sha256_file(_inside(root, name)) == checksum, "immutable historical-freeze input changed")
    prior = {name: _json(root / f"prior/{name}.json") for name in ("B", "C", "POD", "fifth")}
    clean_audit = _json(root / "prior_guard_audit.json")
    _require(set(frozen["prior_receipt_sha256"]) == {"B", "C", "POD", "fifth"} and
             clean_audit["prior_receipt_sha256"] == frozen["prior_receipt_sha256"] and
             clean_audit["old_fifth_provenance"] == prior["fifth"]["provenance"] and
             clean_audit["runtime_contract"] == frozen["runtime_contract"] and
             all(sha256_file(root / f"prior/{name}.json") == checksum for name, checksum in frozen["prior_receipt_sha256"].items()) and
             prior["C"]["selection_freeze_sha256"] == prior["POD"]["selection_freeze_sha256"] == prior["fifth"]["selection_freeze_sha256"] == frozen["prior_receipt_sha256"]["B"] and
             prior["fifth"]["calibration_freeze_sha256"] == frozen["prior_receipt_sha256"]["C"] and
             prior["fifth"]["pod_freeze_sha256"] == frozen["prior_receipt_sha256"]["POD"], "historical prior freeze chain changed")
    for model in frozen["models"]:
        expected_kind = "fno" if model["label"] in LABELS[:3] else ("pod" if model["label"] == pod.LABEL else "physical_coarse_adjoint")
        _require(model["kind"] == expected_kind, "historical model backend differs from its fixed label")
        expected = (next(item["checkpoint_sha256"] for item in prior["B"]["models"] if item["label"] == model["label"])
                    if model["kind"] == "fno" else prior["fifth"]["basis_sha256"])
        _require(sha256_file(_inside(root, model["path"])) == model["sha256"] == expected, "historical model weights/basis differ from their original freezes")
    _require(_json(root / "source_admission.json") == frozen["source_admission"] and
             frozen["source_admission"]["explicit_phase_source_files"] ==
             {name: frozen["provenance"]["source_files"][name] for name in sorted(PHASE_FILES)} and
             frozen["source_admission"]["old_recorded_source_files_unchanged"] is True and
             frozen["operator_signature"] == prior["fifth"]["operator_signature"], "historical explicit source/operator admission changed")
    confirmation.verify_prior_stage_binding(frozen["provenance"], frozen["runtime_contract"], repository_root)
    return frozen


def load_historical_arrays(protocol, dataset_dir, metadata):
    """Only the approved outcome runner calls this, after its freeze gates."""
    roles = {}
    geometry = protocol["geometry"]
    receivers = SurveyGeometry.from_grid(tuple(geometry["shape"]), n_sources=6, n_receivers=20,
        source_depth=2, receiver_depth=2, margin=2).receiver_indices
    for role, cfg in protocol["dataset"]["roles"].items():
        path = _inside(dataset_dir, cfg["arrays_file"])
        _require(sha256_file(path) == cfg["arrays_sha256"], "historical cached arrays changed before loading")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        count, shape = len(metadata[role]), tuple(geometry["shape"])
        _require(set(arrays) == {"models", "source_indices", "frequencies_hz", "wavefields", "receiver_indices", "shape", "spacing"} and
                 arrays["models"].shape == (count, *shape) and arrays["wavefields"].shape == (count, *shape, 2) and
                 tuple(arrays["shape"].tolist()) == shape and float(arrays["spacing"]) == geometry["spacing"] and
                 tuple(arrays["receiver_indices"].tolist()) == receivers and
                 arrays["source_indices"].shape == arrays["frequencies_hz"].shape == (count,), "historical arrays/acquisition shape changed")
        _require(np.all(np.isfinite(arrays["models"])) and np.all(arrays["models"] > 0) and np.all(np.isfinite(arrays["wavefields"])) and
                 arrays["source_indices"].tolist() == [int(row["source_index"]) for row in metadata[role]] and
                 arrays["frequencies_hz"].tolist() == [float(row["frequency_hz"]) for row in metadata[role]],
                 "historical arrays and row metadata disagree or contain invalid fields")
        grouped = defaultdict(list)
        for index, row in enumerate(metadata[role]):
            grouped[row["slice_id"]].append(index)
        for indices in grouped.values():
            _require(all(np.array_equal(arrays["models"][indices[0]], arrays["models"][index]) for index in indices),
                     "all24 blocks must share the identical physical slice model")
        roles[role] = (arrays, grouped)
    return roles


def _base_model(identity, objective, gradient, exact_objective, exact_gradient, *, cost):
    return {**identity, "n_blocks": 24, "aggregate_exact_objective": exact_objective,
        "aggregate_approximate_objective": objective,
        **{key.replace("neural", "approximate"): value for key, value in confirmation.aggregate_gradient_metrics(exact_gradient, gradient).items()},
        "certificate_available": None, "certified_descent": None, "exact_fallback_required": None,
        "aggregate_approximate_directional_derivative": None, "aggregate_exact_directional_derivative": None,
        "aggregate_deterministic_bound": None, "aggregate_certified_upper_derivative": None,
        "aggregate_deterministic_covers": None, "aggregate_deterministic_false_acceptance": None,
        "secondary_mean_block_gradient_cosine": None, "aggregate_vs_block_sum_relative_gradient_error": None,
        "online_cost_json": json.dumps(cost, sort_keys=True, allow_nan=False), "source_calibrated_gate": "not evaluated in historical development-transfer phase"}


def _block_row(identity, block, value, gradient, exact_value, exact_gradient, direction):
    metrics = confirmation.aggregate_gradient_metrics(exact_gradient, gradient)
    signed, exact_signed = float(np.dot(gradient, direction)), float(np.dot(exact_gradient, direction))
    _require(math.isfinite(value) and math.isfinite(signed) and math.isfinite(exact_signed), "nonfinite historical block objective/derivative")
    return {**identity, "source_index": int(block.source_index), "frequency_hz": float(block.frequency_hz),
        "exact_objective": exact_value, "approximate_objective": value, "gradient_cosine": metrics["aggregate_gradient_cosine"],
        "exact_gradient_norm": metrics["aggregate_exact_gradient_norm"], "approximate_gradient_norm": metrics["aggregate_neural_gradient_norm"],
        "exact_directional_derivative": exact_signed, "approximate_directional_derivative": signed,
        "realized_directional_error": abs(signed - exact_signed), "certificate_available": None,
        "deterministic_bound": None, "certified_upper_derivative": None, "deterministic_covers": None,
        "correction_applied": None, "correction_skipped": None, "certificate_diagnostics_json": None}


def evaluate_uncertified_model(surrogate, kind, physics, model, survey, receivers, reference, identity, *, accounting=None):
    """Actual aggregate derivative, with separately paid block diagnostics."""
    before = surrogate.cost_snapshot() if kind == "pod" else None
    accounting = accounting if accounting is not None else {}
    values, gradients = [], []
    for block in survey:
        if kind == "fno":
            accounting["block_gradient_calls_attempted"] = accounting.get("block_gradient_calls_attempted", 0) + 1
        value, gradient, _, _ = surrogate.objective_and_gradient(model, physics.source(block.source_index),
            block.observed, receivers, block.frequency_hz)
        if kind == "fno":
            accounting["block_gradient_calls_completed"] = accounting.get("block_gradient_calls_completed", 0) + 1
        values.append(value)
        gradients.append(gradient)
    summed = np.sum(gradients, axis=0)
    if kind == "fno":
        accounting["shared_batch_calls_attempted"] = accounting.get("shared_batch_calls_attempted", 0) + 1
        value, gradient, _, _ = surrogate.survey_objective_and_gradient(model, survey, receivers)
        accounting["shared_batch_calls_completed"] = accounting.get("shared_batch_calls_completed", 0) + 1
        cost = {"proposal_shared_batch_forward_vjp_calls": 1, "proposal_shared_batch_examples": len(survey),
                "offline_single_block_gradient_diagnostic_calls": len(survey), "total_neural_block_examples": 2 * len(survey)}
        agreement = float(np.linalg.norm(gradient - summed) / max(np.linalg.norm(summed), 1e-15))
        _require(agreement <= 2e-4 and math.isclose(value, sum(values), rel_tol=5e-6, abs_tol=1e-10), "historical shared-input FNO aggregate differs from its block sum")
    else:
        value, gradient = float(sum(values)), summed
        after = surrogate.cost_snapshot()
        cost = {key: int(after.get(key, 0) - before.get(key, 0)) for key in after if key not in {"full_dimension", "reduced_dimension"}}
        agreement = 0.0
    norm = float(np.linalg.norm(gradient))
    _require(math.isfinite(norm) and np.all(np.isfinite(gradient)), "nonfinite historical aggregate proposal")
    direction = -gradient / norm if norm > 1e-14 else np.zeros_like(gradient)
    exact = np.sum([entry[1] for entry in reference.values()], axis=0)
    exact_value = float(sum(entry[0] for entry in reference.values()))
    rows = [_block_row(identity, block, value, block_gradient, *reference[block.source_index, block.frequency_hz], direction)
            for block, value, block_gradient in zip(survey, values, gradients, strict=True)]
    row = _base_model(identity, value, gradient, exact_value, exact, cost=cost)
    row.update(aggregate_vs_block_sum_relative_gradient_error=agreement,
        secondary_mean_block_gradient_cosine=float(np.mean([r["gradient_cosine"] for r in rows])),
        aggregate_approximate_directional_derivative=float(np.dot(gradient, direction)),
        aggregate_exact_directional_derivative=float(np.dot(exact, direction)))
    return rows, row, exact, gradient, direction


def evaluate_corrected_result(result, model, signature, survey, reference, identity):
    """All24 common-direction block bounds; keep ordinary unavailable cases."""
    coarse.validate_result_context(result.to_dict(), model, signature, {block.frequency_hz for block in survey})
    _require(result.approximate_directional_derivative is not None and math.isfinite(result.approximate_directional_derivative) and
             not (result.available is False and all(block.available for block in result.block_certificates)),
             "numeric aggregate abstention: preserve raw result and mark historical attempt incomplete")
    gradient, direction = np.asarray(result.approximate_gradient), np.asarray(result.direction)
    exact = np.sum([entry[1] for entry in reference.values()], axis=0)
    _require(gradient.shape == direction.shape == exact.shape and np.all(np.isfinite(gradient)) and np.all(np.isfinite(direction)) and
             result.approximate_objective is not None and math.isfinite(result.approximate_objective), "nonfinite historical physical aggregate")
    norm = float(np.linalg.norm(gradient))
    _require(math.isfinite(norm) and np.allclose(direction, -gradient / norm if norm > 1e-14 else np.zeros_like(gradient), rtol=1e-10, atol=1e-12),
             "historical corrected certificate must use the normalized negative24-block SUM-gradient direction")
    indexed = {(block.source_index, block.frequency_hz): block for block in survey}
    _require(len(indexed) == len(survey) == len(result.block_certificates) == 24 and
             {(block.source_index, block.frequency_hz) for block in result.block_certificates} == set(indexed), "historical corrected certificate omitted or duplicated blocks")
    rows, block_gradients, objectives, bounds = [], [], [], []
    for certificate in result.block_certificates:
        key = (certificate.source_index, certificate.frequency_hz)
        proposal = np.asarray(certificate.physical_gradient)
        row = _block_row(identity, indexed[key], certificate.approximate_objective, proposal, *reference[key], direction)
        _require(certificate.approximate_directional_derivative is not None and
                 math.isclose(row["approximate_directional_derivative"], certificate.approximate_directional_derivative, rel_tol=1e-10, abs_tol=1e-11),
                 "historical block derivative does not use the common physical direction")
        if certificate.available:
            bound = certificate.directional_error_bound
            _require(bound is not None and math.isfinite(bound) and bound >= 0, "available historical block requires a finite nonnegative bound")
            bounds.append(bound)
        else:
            _require(certificate.directional_error_bound is None, "unavailable historical block must keep null bound")
            bound = None
        row.update(certificate_available=bool(certificate.available), deterministic_bound=bound,
            certified_upper_derivative=row["approximate_directional_derivative"] + bound if bound is not None else None,
            deterministic_covers=bool(row["realized_directional_error"] <= bound + 1e-12) if bound is not None else None,
            correction_applied=bool(certificate.correction_applied), correction_skipped=bool(certificate.correction_skipped),
            certificate_diagnostics_json=json.dumps(certificate.to_dict(), sort_keys=True, allow_nan=False))
        rows.append(row)
        block_gradients.append(proposal)
        objectives.append(certificate.approximate_objective)
    _require(np.allclose(np.sum(block_gradients, axis=0), gradient, rtol=1e-10, atol=1e-11) and
             math.isclose(sum(objectives), result.approximate_objective, rel_tol=1e-12, abs_tol=1e-11), "historical physical aggregate is not its actual24-block sum")
    available = all(block.available for block in result.block_certificates)
    _require(result.available is available and result.exact_fallback_required is (not result.certified_descent), "historical aggregate availability/fallback is inconsistent")
    signed, exact_signed = float(np.dot(gradient, direction)), float(np.dot(exact, direction))
    _require(math.isclose(result.approximate_directional_derivative, signed, rel_tol=1e-10, abs_tol=1e-11), "historical aggregate signed derivative differs")
    if available:
        bound = float(sum(bounds))
        upper = signed + bound
        _require(math.isfinite(bound) and math.isclose(result.directional_error_bound, bound, rel_tol=1e-12, abs_tol=1e-11) and
                 math.isclose(result.certified_upper_derivative, upper, rel_tol=1e-12, abs_tol=1e-11) and
                 result.certified_descent is (upper < 0), "historical aggregate bound/strict certificate is inconsistent")
    else:
        bound = upper = None
        _require(result.directional_error_bound is None and result.certified_upper_derivative is None and not result.certified_descent,
                 "historical unavailable aggregate cannot claim a finite certificate")
    _require(all(result.costs["reference_solver"].get(key, 0) == 0 for key in ("factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs", "stability_rhs", "dense_inverse_rhs")),
             "historical online corrected method unexpectedly used a fine solver")
    row = _base_model(identity, result.approximate_objective, gradient, float(sum(entry[0] for entry in reference.values())), exact, cost=result.costs)
    row.update(certificate_available=available, certified_descent=bool(result.certified_descent), exact_fallback_required=bool(result.exact_fallback_required),
        aggregate_approximate_directional_derivative=signed, aggregate_exact_directional_derivative=exact_signed,
        aggregate_deterministic_bound=bound, aggregate_certified_upper_derivative=upper,
        aggregate_deterministic_covers=bool(abs(signed - exact_signed) <= bound + 1e-12) if available else None,
        aggregate_deterministic_false_acceptance=bool(result.certified_descent and exact_signed >= 0),
        secondary_mean_block_gradient_cosine=float(np.mean([r["gradient_cosine"] for r in rows])),
        aggregate_vs_block_sum_relative_gradient_error=float(np.linalg.norm(np.sum(block_gradients, axis=0) - gradient) / max(norm, 1e-15)))
    return rows, row, exact, gradient, direction


def summarize_historical_transfer(blocks, models, protocol):
    """Descriptive parent summaries only; slices/blocks are not independent n."""
    expected = {(role, slice_id, label): parent for role, cfg in protocol["dataset"]["roles"].items()
                for parent, slices in cfg["parents"].items() for slice_id in slices for label in LABELS}
    _require(len(models) == len(expected) == 100 and len(blocks) == 2400 and
             {(r["role"], r["slice_id"], r["checkpoint"]) for r in models} == set(expected), "historical completion requires all20 slices and all5 models")
    grouped = defaultdict(list)
    for row in blocks:
        key = (row["role"], row["slice_id"], row["checkpoint"])
        _require(key in expected and row["model_id"] == expected[key], "historical block parent/model identity changed")
        grouped[key].append(row)
    pairs = {(source, frequency) for source in protocol["geometry"]["source_indices"] for frequency in protocol["geometry"]["frequencies_hz"]}
    _require(set(grouped) == set(expected) and all(len(group) == 24 and {(r["source_index"], r["frequency_hz"]) for r in group} == pairs for group in grouped.values()),
             "historical results must retain all24 correlated blocks for every slice/model")
    parent_rows, roles = [], {}
    for role, cfg in protocol["dataset"]["roles"].items():
        roles[role] = {}
        for label in LABELS:
            selected = [r for r in models if r["role"] == role and r["checkpoint"] == label]
            per_parent = []
            for parent, slices in cfg["parents"].items():
                group = [r for r in selected if r["model_id"] == parent]
                _require(len(group) == len(slices) and {r["slice_id"] for r in group} == set(slices) and
                         all(np.isfinite(r["aggregate_gradient_cosine"]) for r in group), "historical parent summary omitted/changed a slice")
                row = {"role": role, "checkpoint": label, "model_id": parent, "n_slices": len(group),
                    "mean_actual_aggregate_gradient_cosine": float(np.mean([r["aggregate_gradient_cosine"] for r in group])),
                    "true_descent_slice_fraction": float(np.mean([r["aggregate_approximate_true_descent"] for r in group])),
                    "certificate_available_slice_fraction": None, "certified_descent_slice_fraction": None,
                    "fallback_required_slices": None, "false_accepted_slices": None, "available_bound_failures": None}
                if label == coarse.LABEL:
                    row.update(certificate_available_slice_fraction=float(np.mean([r["certificate_available"] for r in group])),
                        certified_descent_slice_fraction=float(np.mean([r["certified_descent"] for r in group])),
                        fallback_required_slices=sum(r["exact_fallback_required"] for r in group),
                        false_accepted_slices=sum(r["aggregate_deterministic_false_acceptance"] for r in group),
                        available_bound_failures=sum(r["certificate_available"] and not r["aggregate_deterministic_covers"] for r in group))
                else:
                    _require(all(r["certificate_available"] is None and r["certified_descent"] is None for r in group),
                             "uncertified historical baseline cannot acquire a new statistical/deterministic gate")
                per_parent.append(row)
                parent_rows.append(row)
            means = [r["mean_actual_aggregate_gradient_cosine"] for r in per_parent]
            roles[role][label] = {"n_parent_groups": len(per_parent), "n_slices": sum(r["n_slices"] for r in per_parent),
                "equal_parent_mean_actual_aggregate_gradient_cosine": float(np.mean(means)),
                "descriptive_parent_cosine_range": [min(means), max(means)],
                "equal_parent_mean_true_descent_slice_fraction": float(np.mean([r["true_descent_slice_fraction"] for r in per_parent])),
                "parent_cosines": {r["model_id"]: r["mean_actual_aggregate_gradient_cosine"] for r in per_parent},
                "scope": "Historical test reuse/development transfer; descriptive fixed parents only, no IID intervals, nominal conformal or pristine-confirmation claim."}
    _require(len(parent_rows) == 45, "historical report must retain all9 parent groups for all5 models")
    return roles, parent_rows


def run_historical_transfer(*, freeze_dir, repository_root, dataset_dir, output_dir):
    """Requires separate root approval; all model/data/source gates run first."""
    import torch

    frozen = validate_historical_freeze(freeze_dir, repository_root)
    protocol, metadata, parent_audit = validate_historical_protocol(Path(freeze_dir) / "protocol.json", repository_root, dataset_dir=dataset_dir)
    # No project test arrays or model predictions before all freeze/metadata checks.
    roles = load_historical_arrays(protocol, dataset_dir, metadata)
    output = confirmation._new_directory(output_dir)
    torch.set_num_threads(1)
    provenance = environment_record(repository_root)
    bindings = {str((Path(freeze_dir) / "historical_freeze.json").resolve()): sha256_file(Path(freeze_dir) / "historical_freeze.json")}
    for cfg in protocol["dataset"]["roles"].values():
        for name, checksum in ((cfg["arrays_file"], cfg["arrays_sha256"]), (cfg["metadata_file"], cfg["metadata_sha256"])):
            bindings[str(_inside(dataset_dir, name))] = checksum
    bindings[str((Path(dataset_dir) / "dataset_manifest.json").resolve())] = protocol["dataset"]["dataset_manifest_sha256"]
    bindings[str((Path(dataset_dir) / "split_manifest.csv").resolve())] = protocol["dataset"]["role_manifest_sha256"]
    status = {"status": "running", "started_utc": confirmation._utc(), "completed_blocks": 0,
        "scope": "HISTORICAL TEST REUSE/development transfer", "input_sha256": bindings, "provenance": provenance}
    status_path = output / "run_status.json"
    status_path.write_bytes(confirmation._json_bytes(status))
    blocks, models, exact_vectors, proposed_vectors, directions, reference_costs = [], [], [], [], [], []
    neural_accounting = {label: {} for label in LABELS[:3]}
    current = None
    try:
        fno_models = {model["label"]: TrainedFNOWavefieldSurrogate.load(_inside(freeze_dir, model["path"]), device="cpu")
                      for model in frozen["models"] if model["kind"] == "fno"}
        _require(all(model.shape == (24, 24) for model in fno_models.values()), "historical FNO grids must remain24x24")
        basis = _inside(freeze_dir, next(model["path"] for model in frozen["models"] if model["label"] == pod.LABEL))
        raw_pod = PODGalerkinSurrogate.load(basis, rank=64)
        _, corrected_pod, stability, certifier = coarse._make_components(basis)
        _require(stability.operator_signature == frozen["operator_signature"], "historical coarse operator signature changed")
        physics = Helmholtz2D(shape=(24, 24), spacing=.05, damping_width=3, damping_strength=2.)
        for role in ("test_id", "test_ood"):
            arrays, grouped = roles[role]
            receivers = tuple(arrays["receiver_indices"].tolist())
            for slice_id, indices in sorted(grouped.items()):
                parent = metadata[role][indices[0]]["model_id"]
                current = {"role": role, "slice_id": slice_id, "model_id": parent}
                velocity = 1.0 / np.sqrt(arrays["models"][indices[0]].astype(float))
                smooth = ndi.gaussian_filter(velocity, sigma=1.6, mode="reflect")
                model = (1.0 / np.maximum(smooth, 1e-8)**2).reshape(-1)
                physics.clear_solver_cache()
                before = physics.cost_snapshot()
                survey, reference = [], {}
                for index in indices:
                    source, frequency = int(arrays["source_indices"][index]), float(arrays["frequencies_hz"][index])
                    field = arrays["wavefields"][index]
                    observed = (field[..., 0] + 1j * field[..., 1]).reshape(-1)[list(receivers)].astype(np.complex128)
                    value, gradient, _, _ = physics.objective_and_gradient(model, physics.source(source), observed, receivers, frequency)
                    survey.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
                    reference[source, frequency] = (value, gradient)
                after = physics.cost_snapshot()
                reference_costs.append({**current, **{key: int(after.get(key, 0) - before.get(key, 0)) for key in after}})
                confirmation._write_rows(output / "exact_reference_costs.csv", reference_costs)
                for label in LABELS:
                    identity = {**current, "checkpoint": label}
                    current = identity
                    if label == coarse.LABEL:
                        result = certifier.certify_survey(model, survey, receivers, method="coarse_corrected_oriented")
                        confirmation._write_once(output, f"certificates/{role}/{slice_id}.json", confirmation._json_bytes(result.to_dict()))
                        rows, row, exact, proposed, direction = evaluate_corrected_result(result, model, frozen["operator_signature"], survey, reference, identity)
                    else:
                        surrogate, kind = (raw_pod, "pod") if label == pod.LABEL else (fno_models[label], "fno")
                        rows, row, exact, proposed, direction = evaluate_uncertified_model(surrogate, kind, physics, model, survey, receivers, reference, identity,
                            accounting=neural_accounting[label] if kind == "fno" else None)
                    blocks.extend(rows)
                    models.append(row)
                    exact_vectors.append(exact)
                    proposed_vectors.append(proposed)
                    directions.append(direction)
                    confirmation._write_rows(output / "blocks.csv", blocks)
                    confirmation._write_rows(output / "models.csv", models)
                    status.update(completed_blocks=len(blocks), last_completed_case=current,
                        fine_exact_reference_cost=physics.cost_snapshot(), raw_pod_reduced_work=raw_pod.cost_snapshot(),
                        corrected_online_proposal_certificate_cost=certifier.cost_snapshot(), neural_api_call_accounting=neural_accounting)
                    status_path.write_bytes(confirmation._json_bytes(status))
        grouped, parents = summarize_historical_transfer(blocks, models, protocol)
        confirmation._write_rows(output / "parents.csv", parents)
        confirmation._write_rows(output / "exact_reference_costs.csv", reference_costs)
        validate_historical_freeze(freeze_dir, repository_root)
        _require(source_digest(repository_root) == provenance["source_sha256"] and all(sha256_file(path) == checksum for path, checksum in bindings.items()),
                 "historical source or frozen inputs changed during evaluation")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, exact_gradients=np.asarray(exact_vectors), approximate_gradients=np.asarray(proposed_vectors),
            common_directions=np.asarray(directions), role=np.asarray([r["role"] for r in models]),
            model_id=np.asarray([r["model_id"] for r in models]), slice_id=np.asarray([r["slice_id"] for r in models]),
            checkpoint=np.asarray([r["checkpoint"] for r in models]))
        confirmation._write_once(output, "aggregate_gradients.npz", buffer.getvalue())
        summary = {"schema_version": 1, "status": "complete", "completed_utc": confirmation._utc(), "scope": "HISTORICAL TEST REUSE/development transfer; not pristine confirmation",
            "n_slice_model_aggregates": len(models), "n_block_model_rows": len(blocks), "n_parent_model_summaries": len(parents),
            "roles": grouped, "parent_role_audit": parent_audit, "provenance": provenance, "runtime_contract": confirmation._runtime_contract(provenance),
            "input_sha256": bindings, "source_unchanged_during_run": True, "input_files_unchanged_during_run": True,
            "aggregate_vectors_sha256": sha256_file(output / "aggregate_gradients.npz"),
            "fine_exact_reference_cost": physics.cost_snapshot(), "raw_pod_reduced_work": raw_pod.cost_snapshot(),
            "neural_api_call_accounting": neural_accounting,
            "raw_pod_operator_projection_cost": raw_pod.offline_projection_cost,
            "corrected_online_proposal_certificate_cost": certifier.cost_snapshot(),
            "corrected_pod_operator_projection_cost": corrected_pod.offline_projection_cost,
            "cost_scope": "All current reference/proposal/diagnostic work recorded; original observations/training/SVD/calibration are prior paid work, not newly measured free computation. No inversion or acceleration benchmark.",
            "online_exact_recovery_performed": False, "learning_reselection_recalibration_performed": False,
            "statistical_scope": "Fixed historical parent summaries:8 ID parents/16 slices and1 OOD parent/4 slices separately. No IID intervals, slice/trajectory guarantee, nominal conformal, geological population or unseen-test claim."}
        confirmation._write_once(output, "summary.json", confirmation._json_bytes(summary))
        status.update(status="complete", completed_blocks=len(blocks), source_unchanged_during_run=True, input_files_unchanged_during_run=True)
        status_path.write_bytes(confirmation._json_bytes(status))
        return summary
    except Exception as error:
        status.update(status="failed", completed_blocks=len(blocks), failed_case=current, error_type=type(error).__name__, error=str(error))
        status["neural_api_call_accounting"] = neural_accounting
        for name, obj in (("fine_exact_reference_cost", locals().get("physics")), ("raw_pod_reduced_work", locals().get("raw_pod")),
                          ("corrected_online_proposal_certificate_cost", locals().get("certifier"))):
            if obj is not None:
                status[name] = obj.cost_snapshot()
        status_path.write_bytes(confirmation._json_bytes(status))
        raise
