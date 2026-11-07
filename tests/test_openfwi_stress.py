import copy
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from wavecert.experiments.openfwi_stress import (
    prepare_external_models,
    summarize_stress_rows,
    validate_openfwi_stress_inputs,
)
from wavecert.provenance import sha256_file


def test_external_preprocessing_preserves_units_and_values_outside_legacy_clip():
    for meters_per_second in (800.0, 6000.0):
        truth, initial = prepare_external_models(np.full((70, 70), meters_per_second))
        expected = 1 / (meters_per_second * 0.001)**2
        assert truth.shape == initial.shape == (576,)
        assert np.allclose(truth, expected)
        assert np.allclose(initial, expected)
    with pytest.raises(ValueError):
        prepare_external_models(np.zeros((70, 70)))
    with pytest.raises(ValueError):
        prepare_external_models(np.ones((1, 70, 70)))


def _fixture(root):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/openfwi_external_stress.json").read_text())
    data = root / "data"
    data.mkdir()
    (root / "configs").mkdir()
    prepared, calibrations, selections = [], [], {}
    for family in config["families"]:
        family["expected_shape"] = [500, 70, 70]
        values = np.arange(500, dtype=np.float32) + 1000 + 1000 * family["family_code"]
        batch = np.broadcast_to(values[:, None, None], (500, 70, 70)).copy()
        path = data / f"{family['name']}.npy"
        np.save(path, batch)
        raw = path.read_bytes()
        family["expected_uncompressed_bytes"] = len(raw)
        family["expected_crc32"] = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
        indices = sorted(np.random.default_rng(np.random.SeedSequence([
            config["selection"]["seed"], family["family_code"],
        ])).choice(500, size=100, replace=False).tolist())
        selections[family["name"]] = indices
        prepared.append({
            "name": family["name"], "family_code": family["family_code"], "member": family["member"],
            "path": path.name, "sha256": sha256_file(path), "shape": [500, 70, 70], "dtype": "float32",
            "selected_indices": indices,
            "selected_model_ids": [f"{family['name']}:{family['member']}:{i}" for i in indices],
            "selected_model_sha256": [hashlib.sha256(batch[i].tobytes()).hexdigest() for i in indices],
            "duplicate_hash_check": {"original_batch_unique_models": 500, "selected_unique_models": 100},
        })
    for checkpoint in config["checkpoints"]:
        path = root / checkpoint["path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(checkpoint["label"].encode())  # integrity-only; no model is loaded
        checkpoint["sha256"] = sha256_file(path)
        calibration = data / f"{checkpoint['label']}.json"
        calibration.write_text(json.dumps({"checkpoint_sha256": checkpoint["sha256"], "conformal_scale": 0.001}))
        checkpoint["calibration_summary_sha256"] = sha256_file(calibration)
        calibrations.append({"label": checkpoint["label"], "path": calibration.name,
            "source_path": checkpoint["calibration_summary_path"], "sha256": sha256_file(calibration)})
    protocol = root / "configs/openfwi_external_stress.json"
    protocol.write_text(json.dumps(config))
    (data / "protocol.json").write_bytes(protocol.read_bytes())
    record = {"protocol_sha256": sha256_file(protocol), "selected_indices_before_payload": selections}
    (data / "protocol_record.json").write_text(json.dumps(record))
    (data / "preparation_manifest.json").write_text(json.dumps({
        "status": "complete", **record, "families": prepared, "frozen_calibration_summaries": calibrations,
    }))
    (root / "configs/openfwi_external_stress_analysis_addendum.json").write_text(json.dumps({
        "schema_version": 1, "protocol_sha256": sha256_file(protocol), "confidence": 0.95,
    }))
    return protocol, data


def test_frozen_input_validation_rejects_changed_selection_checkpoint_calibration_and_payload(tmp_path):
    protocol, data = _fixture(tmp_path)

    def validate():
        return validate_openfwi_stress_inputs(protocol_path=protocol, data_dir=data, repository_root=tmp_path)

    valid = validate()
    assert len(valid.families) == 2
    assert all(c["intervals_eligible_under_declared_no_duplicate_design"] for c in valid.uncertainty_checks.values())
    manifest_path = data / "preparation_manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["families"][0]["selected_indices"][0] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selected indices"):
        validate()
    manifest_path.write_bytes(original)
    for path, message in ((next(iter(valid.checkpoint_paths.values())), "checkpoint SHA256"),
                          (data / "forward_only.json", "calibration summary SHA256"),
                          (data / "FlatVel_A.npy", "payload SHA256")):
        previous = path.read_bytes()
        path.write_bytes(previous + b" ")
        with pytest.raises(ValueError, match=message):
            validate()
        path.write_bytes(previous)


def _rows_and_protocol():
    protocol = {"families": [{"name": f, "expected_n_models": 5, "selected_n_models": 2} for f in ("a", "b")],
        "checkpoints": [{"label": c} for c in ("forward_only", "derivative_informed")],
        "analysis": {"confidence": 0.95, "bootstrap_resamples": 2000, "bootstrap_seed": 2026110712}}
    rows = []
    for family in ("a", "b"):
        for checkpoint in ("forward_only", "derivative_informed"):
            for model in range(2):
                for source, frequency in ((1, 3.0), (1, 4.0), (2, 3.0), (2, 4.0)):
                    fail = family == "a" and checkpoint == "forward_only" and model == 0 and source == 1 and frequency == 3.0
                    rows.append({"family": family, "checkpoint": checkpoint, "model_id": f"{family}:{model}",
                        "original_index": model, "raw_model_sha256": f"{family}:{model}",
                        "source_index": source, "frequency_hz": frequency,
                        "forward_receiver_relative_l2": 0.6, "gradient_cosine": 0.2,
                        "neural_true_descent": not fail, "deterministic_covers": True,
                        "source_calibrated_covers": not fail, "deterministic_accepts": False,
                        "source_calibrated_accepts": fail, "deterministic_false_acceptance": False,
                        "source_calibrated_false_acceptance": fail})
    return rows, protocol


def test_stress_uncertainty_groups_all_four_blocks_without_pooling_environments():
    rows, protocol = _rows_and_protocol()
    checks = {f: {"intervals_eligible_under_declared_no_duplicate_design": True} for f in ("a", "b")}
    models, summary = summarize_stress_rows(rows, protocol, checks)
    assert len(models) == 8
    group = summary["a"]["forward_only"]
    assert group["n_original_generated_models"] == 2
    assert group["source_calibrated_block_coverage"] == 7/8
    assert group["model_all_blocks_source_calibrated_coverage"] == 1/2
    event = group["model_event_uncertainty"]["model_any_source_calibrated_false_acceptance"]
    assert event["fixed_batch_uniform_sample"]["observed_failures"] == 1
    assert event["fixed_batch_uniform_sample"]["sample_size"] == 2
    assert event["conditional_generated_draw_binomial"]["n_units"] == 2
    assert summary["b"]["forward_only"]["model_all_blocks_source_calibrated_coverage"] == 1
    changed = copy.deepcopy(checks)
    changed["a"]["intervals_eligible_under_declared_no_duplicate_design"] = False
    kept_models, withheld = summarize_stress_rows(rows, protocol, changed)
    assert len(kept_models) == 8
    assert withheld["a"]["forward_only"]["model_event_uncertainty"]["model_any_source_calibrated_false_acceptance"]["status"].startswith("unavailable")
    assert withheld["a"]["forward_only"]["conditional_model_bootstrap_mean_gradient_cosine"]["interval"] is None
