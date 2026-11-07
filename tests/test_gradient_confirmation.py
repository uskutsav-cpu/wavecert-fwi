import copy
import csv
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/gradient_confirmation_plan.json"


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def training_provenance():
    revision = subprocess.check_output(["git", "rev-parse", "be6e51e"], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True).splitlines()
    manifest = {name: sha256_file(ROOT / name) for name in sorted(names) if confirmation._training_source_path(name)}
    digest = hashlib.sha256()
    for name, checksum in manifest.items():
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
    return {"git_revision": revision, "source_files": manifest, "source_sha256": digest.hexdigest()}


@pytest.fixture
def completed_training(tmp_path, training_provenance):
    """Fabricated audit receipts only; no learned model or held-out prediction."""
    plan = json.loads(PLAN.read_text())
    protocol = json.loads((ROOT / plan["development_training"]["protocol_path"]).read_text())
    training = tmp_path / "training"
    training.mkdir()
    (training / "protocol.json").write_bytes((ROOT / plan["development_training"]["protocol_path"]).read_bytes())
    inputs = {}
    for role in ("train", "validation"):
        for name in (f"{role}.npz", f"{role}_metadata.csv"):
            relative = f"data/reference/subsurfacegen_parent_split/{name}"
            inputs[relative] = sha256_file(ROOT / relative)
    input_audit = {"input_file_sha256": inputs, "opened_array_roles": ["train", "validation"]}
    _save(training / "input_audit.json", input_audit)
    candidates = []
    for order, candidate in enumerate(protocol["candidates"]):
        directory = training / "candidates" / candidate["name"]
        directory.mkdir(parents=True)
        weight = directory / "best.pt"
        weight.write_bytes(f"audit-fixture-not-a-neural-model:{candidate['name']}".encode())
        score = 4.0 - order
        item = {"name": candidate["name"], "status": "complete", "optimizer_updates_completed": 2000,
                "best_optimizer_update": 100, "best_validation_score": score,
                "checkpoint_path": str(weight), "checkpoint_sha256": sha256_file(weight),
                "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False}
        candidates.append(item)
        _save(directory / "summary.json", item)
        with (directory / "training_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["optimizer_update", "batch_size"])
            writer.writeheader()
            writer.writerows({"optimizer_update": i, "batch_size": 32} for i in range(1, 2001))
        validations = []
        for update in range(0, 2001, 100):
            value = score if update == 100 else score + 1
            validations.append({"optimizer_update": update, "checkpoint_eligible": update > 0,
                "selection_score": value, "receiver_relative_squared_l2": value - .85,
                "jvp_relative_squared_l2": .1, "direction_loss": .75, "aggregate_gradient_cosine": .25})
        _save(directory / "validation_history.json", validations)
    chosen = candidates[-1]
    selection = {"winner": chosen["name"], "checkpoint_sha256": chosen["checkpoint_sha256"],
                 "best_optimizer_update": 100, "validation_selection_score": chosen["best_validation_score"],
                 "source_and_inputs_unchanged": True}
    summary = {"status": "complete", "source_unchanged_during_run": True, "inputs_unchanged_during_run": True,
               "protocol_sha256": sha256_file(training / "protocol.json"), "provenance": training_provenance,
               "source_sha256_at_end": training_provenance["source_sha256"], "candidates": candidates,
               "input_audit": input_audit, "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False}
    _save(training / "summary.json", summary)
    _save(training / "completed_candidates.json", candidates)
    _save(training / "selection.json", selection)
    _save(training / "status.json", {"status": "complete", "protocol_sha256": summary["protocol_sha256"]})
    return training


def test_prospective_fold_is_disjoint_and_uses_the_fixed400_complement():
    plan = confirmation.validate_confirmation_plan(PLAN, ROOT)
    for family in plan["families"]:
        population, selected = confirmation.confirmation_indices(family["excluded_old_indices"], seed=2026110731, family_code=family["family_code"])
        assert len(population) == 400 and len(set(selected)) == 100
        assert set(population).isdisjoint(family["excluded_old_indices"])
        assert set(population) | set(family["excluded_old_indices"]) == set(range(500))
        assert selected == family["confirmation_selected_indices"]


def test_plan_fold_tampering_cannot_move_an_old_test_model_into_confirmation(tmp_path):
    plan = json.loads(PLAN.read_text())
    plan["families"][0]["confirmation_selected_indices"][0] = plan["families"][0]["excluded_old_indices"][0]
    path = tmp_path / "changed.json"
    _save(path, plan)
    with pytest.raises(ValueError, match="fold"):
        confirmation.validate_confirmation_plan(path, ROOT)


def test_selection_freeze_retains_three_models_without_modifying_original_training(completed_training, tmp_path):
    original = {p: sha256_file(p) for p in completed_training.rglob("*") if p.is_file()}
    output = tmp_path / "freeze"
    result = confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
        training_dir=completed_training, training_source_root=ROOT, output_dir=output)
    assert result["selected_candidate"] == "augmented_derivative"
    assert len(result["all_candidate_checkpoints"]) == 4
    assert {m["label"] for m in result["models"]} == {"legacy_forward", "legacy_jvp", "development_winner"}
    confirmation.validate_selection_freeze(output)
    assert all(sha256_file(p) == digest for p, digest in original.items())
    with pytest.raises(ValueError, match="new directory"):
        confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
            training_dir=completed_training, training_source_root=ROOT, output_dir=output)
    (output / "checkpoints/development_winner.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        confirmation.validate_selection_freeze(output)


@pytest.mark.parametrize("change", ["unfinished", "missing_candidate", "winner", "weight", "history"])
def test_selection_freeze_refuses_partial_or_changed_training(completed_training, tmp_path, change):
    if change == "unfinished":
        value = json.loads((completed_training / "status.json").read_text())
        value["status"] = "running"
        _save(completed_training / "status.json", value)
    elif change == "missing_candidate":
        completed = json.loads((completed_training / "completed_candidates.json").read_text())[:-1]
        summary = json.loads((completed_training / "summary.json").read_text())
        summary["candidates"] = completed
        _save(completed_training / "completed_candidates.json", completed)
        _save(completed_training / "summary.json", summary)
    elif change == "winner":
        selection = json.loads((completed_training / "selection.json").read_text())
        selection["winner"] = "raw_forward"
        _save(completed_training / "selection.json", selection)
    elif change == "weight":
        (completed_training / "candidates/augmented_derivative/best.pt").write_bytes(b"changed")
    else:
        path = completed_training / "candidates/raw_forward/training_history.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError):
        confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
            training_dir=completed_training, training_source_root=ROOT, output_dir=tmp_path / "rejected")
    assert not (tmp_path / "rejected/freeze.json").exists()


def test_heldout_tools_cannot_predict_without_the_selection_and_calibration_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation.TrainedFNOWavefieldSurrogate, "load", lambda *a, **k: pytest.fail("prediction model loaded before freeze"))
    with pytest.raises(FileNotFoundError):
        confirmation.run_selected_calibration(freeze_dir=tmp_path / "absent", repository_root=ROOT,
            dataset_dir=ROOT / "data/reference/subsurfacegen_parent_split", output_dir=tmp_path / "calibration")
    with pytest.raises(FileNotFoundError):
        confirmation.run_gradient_confirmation(plan_path=PLAN, repository_root=ROOT,
            selection_dir=tmp_path / "absent", calibration_dir=tmp_path / "absent", data_dir=tmp_path,
            output_dir=tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_selected_calibration_is_gated_and_legacy_scales_remain_pinned(completed_training, tmp_path, monkeypatch):
    from wavecert.experiments import real_wavecert

    frozen_dir, calibration_dir = tmp_path / "selection_freeze", tmp_path / "calibration_freeze"
    frozen = confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
        training_dir=completed_training, training_source_root=ROOT, output_dir=frozen_dir)
    calls = []

    def fake_calibration(**kwargs):
        # No predictions: simulate the existing calibrator's receipt contract.
        assert (frozen_dir / "freeze.json").is_file()
        calls.append(kwargs)
        report = {"calibration_parent_models": 12, "alpha": .1, "conformal_scale": .5,
            "checkpoint_sha256": sha256_file(kwargs["checkpoint_path"]),
            "dataset_provenance": {"dataset_manifest_sha256": json.loads(PLAN.read_text())["selected_calibration"]["dataset_manifest_sha256"]},
            "id": {"n_parent_models": 8}, "ood": {"n_parent_models": 1}}
        _save(Path(kwargs["output_dir"]) / "summary.json", report)
        return report

    monkeypatch.setattr(real_wavecert, "run_real_wavecert_study", fake_calibration)
    result = confirmation.run_selected_calibration(freeze_dir=frozen_dir, repository_root=ROOT,
        dataset_dir=ROOT / "data/reference/subsurfacegen_parent_split", output_dir=calibration_dir)
    assert len(calls) == 1
    assert Path(calls[0]["checkpoint_path"]).name == "development_winner.pt"
    confirmation.validate_calibration_freeze(frozen_dir, calibration_dir)
    for legacy in frozen["models"][:2]:
        selected = next(m for m in result["models"] if m["label"] == legacy["label"])
        assert selected["calibration_sha256"] == legacy["calibration_sha256"]
    # Rewriting both a legacy report and its new receipt hash cannot authorize
    # recalibration: it still has to match the original selection freeze.
    legacy = result["models"][0]
    path = calibration_dir / legacy["calibration_path"]
    changed = json.loads(path.read_text())
    changed["conformal_scale"] += .1
    _save(path, changed)
    legacy["calibration_sha256"] = sha256_file(path)
    _save(calibration_dir / "calibration_freeze.json", result)
    with pytest.raises(ValueError, match="legacy calibration"):
        confirmation.validate_calibration_freeze(frozen_dir, calibration_dir)


def test_actual_aggregate_gradient_is_not_the_mean_block_cosine():
    exact_blocks = np.array([[100., 0], [0, 1.]])
    neural_blocks = np.array([[-1., 0], [0, 100.]])
    # Individual cosines average to zero, and the initial summed gradients
    # are orthogonal in this example.
    individual = [-1., 1.]
    metrics = confirmation.aggregate_gradient_metrics(exact_blocks.sum(axis=0), neural_blocks.sum(axis=0))
    assert np.mean(individual) == 0
    assert metrics["aggregate_gradient_cosine"] == pytest.approx(0., abs=1e-15)
    # Change only the magnitude of the first block: average block cosines is
    # still zero but the summed direction now goes uphill.
    neural_blocks[0, 0] = -10
    metrics = confirmation.aggregate_gradient_metrics(exact_blocks.sum(axis=0), neural_blocks.sum(axis=0))
    assert metrics["aggregate_gradient_cosine"] < -0.08
    assert not metrics["aggregate_neural_true_descent"]
    zero = confirmation.aggregate_gradient_metrics(np.ones(2), np.zeros(2))
    assert zero["aggregate_neural_gradient_zero"] and not zero["aggregate_neural_true_descent"]


def test_regenerated_local_receipt_bytes_need_not_equal_the_published_historical_receipt(tmp_path):
    batch = np.repeat(np.arange(1, 501, dtype=np.float32)[:, None, None, None], 4, axis=3).reshape(500, 1, 2, 2)
    path = tmp_path / "velocity.npy"
    np.save(path, batch)
    checksum = sha256_file(path)
    crc = f"{zlib.crc32(path.read_bytes()) & 0xffffffff:08x}"
    excluded = list(range(100))
    family = {"name": "test", "family_code": 0, "member": "test.npy", "raw_npy_relative_path": "velocity.npy",
              "raw_npy_sha256": checksum, "expected_crc32": crc, "expected_shape": [500, 1, 2, 2],
              "expected_dtype": "float32", "excluded_old_indices": excluded,
              "eligible_complement_indices": list(range(100, 500)), "confirmation_selected_indices": list(range(100, 200))}
    protocol_sha = "fixed-old-protocol"
    local = {"status": "complete", "protocol_sha256": protocol_sha, "created_utc": "new runtime receipt",
             "families": [{"name": "test", "family_code": 0, "member": "test.npy", "sha256": checksum,
                           "selected_indices": excluded, "zip_member": {"crc32": crc}}]}
    _save(tmp_path / "preparation_manifest.json", local)
    plan = {"families": [family], "old_external": {"protocol_sha256": protocol_sha,
                                                    "published_preparation_manifest_sha256": "different historical bytes"}}
    _, hashes, audit = confirmation.validate_confirmation_payloads(plan, tmp_path)
    assert audit["test"]["analysis_population_size"] == 400
    assert hashes[str((tmp_path / "preparation_manifest.json").resolve())] != "different historical bytes"
    changed = copy.deepcopy(plan)
    changed["families"][0]["raw_npy_sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA256"):
        confirmation.validate_confirmation_payloads(changed, tmp_path)


def test_confirmation_summary_uses_N400_and_requires_every_fixed_model_label():
    plan = json.loads(PLAN.read_text())
    blocks, models, audits = [], [], {}
    for family in plan["families"]:
        audits[family["name"]] = {"population_unique_raw_hashes": 400}
        for label in ("legacy_forward", "legacy_jvp", "development_winner"):
            for index in family["confirmation_selected_indices"]:
                models.append({"family": family["name"], "checkpoint": label, "original_index": index,
                               "aggregate_gradient_cosine": .3, "aggregate_neural_true_descent": True})
                for source, frequency in ((1, 3.), (2, 3.), (1, 4.), (2, 4.)):
                    blocks.append({"family": family["name"], "checkpoint": label, "original_index": index,
                        "source_index": source, "frequency_hz": frequency, "gradient_cosine": -.5,
                        "source_calibrated_false_acceptance": False, "source_calibrated_covers": True,
                        "deterministic_false_acceptance": False, "deterministic_covers": True,
                        "source_calibrated_accepts": True})
    summary = confirmation.summarize_confirmation(blocks, models, plan, audits)
    for family in summary.values():
        assert len(family) == 3
        for model in family.values():
            assert model["primary_mean_actual_aggregate_gradient_cosine"] == pytest.approx(.3)
            assert model["block_gradient_cosine_median_secondary"] == -.5
            bound = model["fixed_complement_hypergeometric_bounds"]["aggregate_neural_non_descent"]
            assert bound["population_size"] == 400 and bound["sample_size"] == 100
    with pytest.raises(ValueError, match="exactly"):
        confirmation.summarize_confirmation(blocks[:-1], models, plan, audits)
