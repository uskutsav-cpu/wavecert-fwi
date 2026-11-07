import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.provenance import environment_record, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/gradient_confirmation_plan.json"


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _current_head_source_provenance():
    """Synthetic runs bind the checkout under test, including shallow clones."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True).splitlines()
    manifest = {name: sha256_file(ROOT / name) for name in sorted(names) if confirmation._training_source_path(name)}
    digest = hashlib.sha256()
    for name, checksum in manifest.items():
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
    return {"git_revision": revision, "source_files": manifest, "source_sha256": digest.hexdigest()}


@pytest.fixture(scope="module")
def training_provenance():
    return _current_head_source_provenance()


@pytest.fixture
def synthetic_confirmation_artifacts(monkeypatch, training_provenance):
    """Consistent synthetic plans/receipts; production pins and files stay intact.

    Historical-study reproduction deliberately needs its historical Git objects.
    These unit fixtures instead claim CURRENT HEAD, then exercise the unchanged
    production source/blob, runtime, identity, selection and freeze validators.
    Generated files live under tests/, outside the production source manifest.
    """
    from wavecert.experiments import coarse_pod_confirmation as coarse
    from wavecert.experiments import pod_confirmation as pod

    with tempfile.TemporaryDirectory(prefix=".synthetic-confirmation-", dir=ROOT / "tests") as directory:
        fixture_root = Path(directory)
        published = ROOT / "configs/gradient_confirmation_plan.json"
        plan = json.loads(published.read_text())
        plan["synthetic_test_fixture"] = True
        plan["development_training"]["source_commit"] = training_provenance["git_revision"]
        plan_path = fixture_root / "plan.json"
        _save(plan_path, plan)
        monkeypatch.setattr(confirmation, "FROZEN_PLAN_SHA256", sha256_file(plan_path))

        study = fixture_root / "pod_development"
        shutil.copytree(ROOT / "results/pod_development", study)
        summary = json.loads((study / "summary.json").read_text())
        summary.update(synthetic_test_fixture=True, provenance=copy.deepcopy(training_provenance),
            source_sha256_at_end=training_provenance["source_sha256"],
            scope="SYNTHETIC unit-test copy of archived POD outcomes with current-checkout source binding; no scientific result")
        _save(study / "summary.json", summary)
        pod_path = fixture_root / "pod_addendum.json"
        addendum = json.loads((ROOT / "configs/pod_confirmation_addendum.json").read_text())
        addendum.update(synthetic_test_fixture=True,
            original_three_fno_plan_path=plan_path.relative_to(ROOT).as_posix(),
            original_three_fno_plan_sha256=sha256_file(plan_path),
            pod_study_path=study.relative_to(ROOT).as_posix(),
            pod_source_commit=training_provenance["git_revision"],
            pod_source_sha256=training_provenance["source_sha256"],
            pod_equivalent_local_cherry_pick_commit=training_provenance["git_revision"],
            pod_source_receipt_scope="SYNTHETIC test metadata, not the historical source-study receipt")
        addendum["pod_study_files_sha256"] = {name: sha256_file(study / name) for name in addendum["pod_study_files_sha256"]}
        _save(pod_path, addendum)
        monkeypatch.setattr(pod, "ADDENDUM_SHA256", sha256_file(pod_path))

        fifth_path = fixture_root / "coarse_addendum.json"
        fifth = json.loads((ROOT / "configs/coarse_corrected_pod_confirmation_addendum.json").read_text())
        fifth.update(synthetic_test_fixture=True,
            original_three_fno_plan={"path": plan_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(plan_path)},
            prior_four_model_pod_addendum={"path": pod_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(pod_path), "preserved_unchanged": True})
        _save(fifth_path, fifth)
        monkeypatch.setattr(coarse, "ADDENDUM_SHA256", sha256_file(fifth_path))
        replacements = {published: plan_path, ROOT / "configs/pod_confirmation_addendum.json": pod_path,
                        ROOT / "configs/coarse_corrected_pod_confirmation_addendum.json": fifth_path}

        historical_path = ROOT / "configs/historical_transfer_protocol.json"
        if historical_path.is_file():
            from wavecert.experiments import historical_transfer as historical

            historical_protocol = json.loads(historical_path.read_text())
            historical_protocol.update(synthetic_test_fixture=True, required_prior_plans={
                Path(os_relative).as_posix(): sha256_file(path) for os_relative, path in (
                    (f"../{plan_path.relative_to(ROOT).as_posix()}", plan_path),
                    (f"../{pod_path.relative_to(ROOT).as_posix()}", pod_path),
                    (f"../{fifth_path.relative_to(ROOT).as_posix()}", fifth_path))})
            synthetic_historical = fixture_root / "historical_protocol.json"
            _save(synthetic_historical, historical_protocol)
            monkeypatch.setattr(historical, "PROTOCOL_SHA256", sha256_file(synthetic_historical))
            replacements[historical_path] = synthetic_historical

        # Test consumers refer to these fixture paths. No validator is mocked:
        # only expected synthetic artifact identities are parameters of the test.
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).resolve().is_relative_to(ROOT / "tests"):
                for name in ("PLAN", "POD_PLAN", "ADDENDUM", "PROTOCOL"):
                    value = getattr(module, name, None)
                    if isinstance(value, Path) and value in replacements:
                        monkeypatch.setattr(module, name, replacements[value])
        yield {"plan": plan_path, "pod_addendum": pod_path, "fifth_addendum": fifth_path,
               "source_provenance": training_provenance}


@pytest.fixture
def completed_training(tmp_path, training_provenance, synthetic_confirmation_artifacts):
    """Fabricated audit receipts only; no learned model or held-out prediction."""
    plan = json.loads(synthetic_confirmation_artifacts["plan"].read_text())
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
    summary = {"status": "complete", "synthetic_test_fixture": True,
               "source_unchanged_during_run": True, "inputs_unchanged_during_run": True,
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
    assert not (tmp_path / "calibration").exists()


def test_fno_outcome_prediction_requires_both_freezes(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(confirmation.TrainedFNOWavefieldSurrogate, "load", lambda *a, **k: pytest.fail("prediction model loaded before freeze"))
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


def _prior_stage(tmp_path):
    source = tmp_path / "src/wavecert/physics/operator.py"
    source.parent.mkdir(parents=True)
    source.write_text("REFERENCE_OPERATOR = 1\n")
    previous = environment_record(tmp_path)
    return source, previous, confirmation._runtime_contract(previous)


def test_stage_binding_rejects_modified_recorded_source_and_allows_new_unused_tool(tmp_path):
    source, previous, runtime = _prior_stage(tmp_path)
    extra = tmp_path / "scripts/unrelated_report.py"
    extra.parent.mkdir()
    extra.write_text("REPORT_ONLY = True\n")
    audit = confirmation.verify_prior_stage_binding(previous, runtime, tmp_path)
    assert audit["new_unimported_source_files"] == ["scripts/unrelated_report.py"]
    source.write_text("REFERENCE_OPERATOR = 2\n")
    with pytest.raises(ValueError, match="source changed.*new immutable freeze"):
        confirmation.verify_prior_stage_binding(previous, runtime, tmp_path)


def test_newly_added_imported_code_is_not_treated_as_an_unrelated_file(tmp_path, monkeypatch):
    _, previous, runtime = _prior_stage(tmp_path)
    extra = tmp_path / "src/wavecert/physics/new_operator.py"
    extra.write_text("NEW_OPERATOR = True\n")
    imported = types.ModuleType("confirmation_future_plugin")
    imported.__file__ = str(extra)
    monkeypatch.setitem(sys.modules, imported.__name__, imported)
    with pytest.raises(ValueError, match="newly added source file is imported"):
        confirmation.verify_prior_stage_binding(previous, runtime, tmp_path)


@pytest.mark.parametrize("component", ["python", "numpy", "scipy", "torch", "device"])
def test_stage_binding_rejects_numerical_runtime_or_cpu_contract_changes(tmp_path, component):
    _, previous, runtime = _prior_stage(tmp_path)
    if component == "python":
        previous["python"] = runtime["python"] = "different-python-version"
    elif component == "device":
        runtime["device"] = "cuda"
    else:
        previous["packages"][component] = runtime["packages"][component] = "different-package-version"
    with pytest.raises(ValueError, match="runtime or CPU device contract changed"):
        confirmation.verify_prior_stage_binding(previous, runtime, tmp_path)


def test_unrelated_optional_package_versions_are_recorded_without_exact_binding(tmp_path):
    _, previous, runtime = _prior_stage(tmp_path)
    previous["packages"]["devito"] = "different-optional-version"
    assert confirmation.verify_prior_stage_binding(previous, runtime, tmp_path)["numerical_runtime_versions_unchanged"]


@pytest.fixture
def recovered_training(completed_training, monkeypatch):
    """An interrupted receipt chain with real torch containers, no predictions."""
    import math
    import shutil

    torch = pytest.importorskip("torch")

    from wavecert.surrogates.fno import WavefieldNormalization

    training = completed_training
    protocol = json.loads((training / "protocol.json").read_text())
    summary = json.loads((training / "summary.json").read_text())
    inputs = summary["input_audit"]
    selected_cal = json.loads(PLAN.read_text())["selected_calibration"]
    inputs.update(dataset_manifest_sha256=selected_cal["dataset_manifest_sha256"],
                  role_manifest_sha256=selected_cal["role_manifest_sha256"])
    _save(training / "input_audit.json", inputs)
    (training / "labels").mkdir()
    models = np.arange(16, dtype=np.float32).reshape(4, 2, 2) + 1
    frequencies = np.array([3., 4., 3., 4.])
    fields = np.arange(32, dtype=np.float32).reshape(4, 2, 2, 2)
    np.savez(training / "labels/train.npz", models=models, frequencies_hz=frequencies, fields=fields)
    for role in ("validation", "validation_inverse_cases"):
        np.savez(training / f"labels/{role}.npz", placeholder=np.array([1.]))
    hashes = {role: sha256_file(training / f"labels/{role}.npz") for role in confirmation.RECOVERY_LABEL_SHA256}
    records = {role: {"artifact_sha256": value} for role, value in hashes.items()}
    _save(training / "teacher_labels.json", records)
    normalization = vars(WavefieldNormalization.from_arrays(models[:2], frequencies[:2], fields[:2]))
    _save(training / "normalization.json", {"source_role": "raw train only", **normalization})
    monkeypatch.setattr(confirmation, "RECOVERY_TEACHER_SHA256", sha256_file(training / "teacher_labels.json"))
    monkeypatch.setattr(confirmation, "RECOVERY_LABEL_SHA256", hashes)
    driver = training / "reviewed_driver.py"
    driver.write_text("# fixture reviewed I/O-only driver\n")
    monkeypatch.setattr(confirmation, "RECOVERY_DRIVER_SHA256", sha256_file(driver))
    for candidate, item in zip(protocol["candidates"], summary["candidates"], strict=True):
        directory = training / "candidates" / candidate["name"]
        validations = json.loads((directory / "validation_history.json").read_text())
        best = validations[1]
        item.update(candidate=candidate, best_validation=best)
        _save(directory / "best_validation.json", best)
        payload = dict(development_source_sha256=summary["source_sha256_at_end"],
            development_protocol_sha256=summary["protocol_sha256"], candidate=candidate,
            optimizer_updates_budget=2000, best_optimizer_update=100,
            best_validation_selection_score=item["best_validation_score"], normalization=normalization,
            model_config=protocol["model"], teacher_label_artifacts=records,
            dataset_manifest_sha256=inputs["dataset_manifest_sha256"], training_manifest_sha256=inputs["role_manifest_sha256"],
            calibration_role_used_for_model_selection=False, calibration_or_test_arrays_loaded=False,
            external_test_arrays_loaded=False)
        torch.save(payload, directory / "best.pt")
        item["checkpoint_sha256"] = sha256_file(directory / "best.pt")
        _save(directory / "summary.json", item)
        keys = ["optimizer_update", "batch_size", "derivative_weight", "jvp_training_evaluated", "total_loss",
                "full_field_relative_squared_l2", "receiver_relative_squared_l2", "preclip_parameter_gradient_norm",
                "learning_rate_used", "jvp_relative_squared_l2"]
        with (directory / "training_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for update in range(1, 2001):
                row = dict.fromkeys(keys, 1.)
                row.update(optimizer_update=update, batch_size=32, derivative_weight=candidate["derivative_weight"],
                    jvp_training_evaluated=candidate["derivative_weight"] > 0,
                    learning_rate_used=protocol["training"]["scheduler_eta_min"] + .5 * (
                        protocol["training"]["learning_rate"] - protocol["training"]["scheduler_eta_min"]) * (
                            1 + math.cos(math.pi * (update - 1) / 2000)))
                writer.writerow(row)
    tag = "fixture-recovery"
    recovery = training / "recovery_attempts" / tag
    restarted = recovery / "candidate/augmented_derivative"
    restarted.parent.mkdir(parents=True)
    partial = training / "candidates/augmented_derivative"
    partial.rename(restarted)
    fourth = summary["candidates"][-1]
    fourth["checkpoint_path"] = str(restarted / "best.pt")
    _save(restarted / "summary.json", fourth)
    partial.mkdir()
    lines = (restarted / "training_history.csv").read_text().splitlines()
    (partial / "training_history.csv").write_text("\n".join(lines[:101]) + "\n")
    _save(partial / "validation_history.json", json.loads((restarted / "validation_history.json").read_text())[:1])
    failure = training / "failure_attempts" / tag
    failure.mkdir(parents=True)
    shutil.copytree(partial, failure / "partial_candidate/augmented_derivative")
    failed = {"status": "failed", "error": "[Errno 28] No space left on device", "protocol_sha256": summary["protocol_sha256"],
              "completed_candidate_names": [c["name"] for c in protocol["candidates"][:3]]}
    _save(failure / "status.json", failed)
    _save(failure / "completed_candidates.json", summary["candidates"][:3])
    protected = {name: sha256_file(training / name) for name in ("protocol.json", "teacher_labels.json", "input_audit.json", "normalization.json")}
    protected.update({f"labels/{role}.npz": value for role, value in hashes.items()})
    for candidate in protocol["candidates"][:3]:
        directory = training / "candidates" / candidate["name"]
        protected.update({p.relative_to(training).as_posix(): sha256_file(p) for p in directory.iterdir()})
    receipt = {"status": "running", "recovery_id": tag, "driver_path": str(driver), "driver_sha256": sha256_file(driver),
        "frozen_source_sha256": summary["source_sha256_at_end"], "protocol_sha256": summary["protocol_sha256"],
        "runtime_audit": {"original_start_runtime_receipt_available": False, "current_environment": summary["provenance"]},
        "original_failure": failed, "protected_file_sha256": protected,
        "preserved_partial_sha256": {p.name: sha256_file(p) for p in partial.iterdir()}}
    _save(failure / "preservation_receipt.json", receipt)
    receipt.update(status="complete", fourth_candidate_restarted_from_seed=True, fourth_failed_attempt_updates_preserved=100,
        fourth_successful_attempt_updates=2000, fourth_total_executed_updates_across_attempts=2100,
        retained_three_candidates_retrained=False, io_policy={"first_100_optimizer_records_identical": True,
        "initial_validation_baseline_identical": True})
    _save(recovery / "recovery_receipt.json", receipt)
    summary.update(recovered_from_interruption=True, source_unchanged_during_run=None, inputs_unchanged_during_run=None,
        source_unchanged_during_recovery=True, inputs_unchanged_during_recovery=True,
        recorded_initial_end_and_recovery_source_digests_match=True, original_start_runtime_receipt_available=False,
        elapsed_seconds=None, initial_source_sha256_recovered_from_retained_checkpoints=summary["source_sha256_at_end"],
        recovery=receipt, teacher_labels=records, normalization=normalization,
        limitations=["Original full start-runtime receipt unavailable; 100 additional updates spent."])
    selection = json.loads((training / "selection.json").read_text())
    selection["checkpoint_sha256"] = fourth["checkpoint_sha256"]
    _save(training / "selection.json", selection)
    _save(training / "completed_candidates.json", summary["candidates"])
    _save(training / "summary.json", summary)
    _save(training / "status.json", {"status": "complete", "protocol_sha256": summary["protocol_sha256"],
                                     "recovery_id": tag, "recovered_from_enospc": True})
    return training


def test_explicit_recovery_admission_preserves_unknowns_and_extra_spent_work(recovered_training, tmp_path):
    original = {p: sha256_file(p) for p in recovered_training.rglob("*") if p.is_file()}
    output = tmp_path / "restart-freeze"
    frozen = confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
        training_dir=recovered_training, training_source_root=ROOT, output_dir=output)
    assert frozen["training_source_and_inputs_unchanged"] is None
    assert frozen["training_admission"]["fourth_candidate_actual_executed_updates"] == 2100
    assert frozen["training_admission"]["total_actual_executed_updates"] == 8100
    assert frozen["training_admission"]["original_start_runtime_receipt_available"] is False
    assert (output / "training/recovery/driver_snapshot.py").read_bytes() == (recovered_training / "reviewed_driver.py").read_bytes()
    confirmation.validate_selection_freeze(output)
    assert all(sha256_file(p) == digest for p, digest in original.items())
    frozen["training_source_and_inputs_unchanged"] = True
    _save(output / "freeze.json", frozen)
    with pytest.raises(ValueError, match="interruption"):
        confirmation.validate_selection_freeze(output)


@pytest.mark.parametrize("change", ["restart_prefix", "initial_baseline", "driver", "protected_history", "runtime", "extra_updates", "nonfinite_loss", "checkpoint_source"])
def test_recovery_admission_rejects_changed_evidence(recovered_training, tmp_path, change):
    import torch

    summary = json.loads((recovered_training / "summary.json").read_text())
    tag = summary["recovery"]["recovery_id"]
    restarted = recovered_training / "recovery_attempts" / tag / "candidate/augmented_derivative"
    if change in {"restart_prefix", "nonfinite_loss"}:
        path = restarted / "training_history.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            keys, rows = reader.fieldnames, list(reader)
        rows[0 if change == "restart_prefix" else 200]["total_loss"] = "7" if change == "restart_prefix" else "nan"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    elif change == "initial_baseline":
        path = restarted / "validation_history.json"
        rows = json.loads(path.read_text())
        rows[0]["selection_score"] += 1
        _save(path, rows)
    elif change == "driver":
        (recovered_training / "reviewed_driver.py").write_text("# unreviewed change\n")
    elif change == "protected_history":
        path = recovered_training / "candidates/raw_forward/training_history.csv"
        path.write_text(path.read_text() + "\n")
    elif change == "runtime":
        summary["original_start_runtime_receipt_available"] = True
        _save(recovered_training / "summary.json", summary)
    elif change == "extra_updates":
        summary["recovery"]["fourth_total_executed_updates_across_attempts"] = 2000
        _save(recovered_training / "summary.json", summary)
        _save(recovered_training / "recovery_attempts" / tag / "recovery_receipt.json", summary["recovery"])
    else:
        payload = torch.load(restarted / "best.pt", weights_only=True)
        payload["development_source_sha256"] = "different source"
        torch.save(payload, restarted / "best.pt")
    with pytest.raises(ValueError):
        confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
            training_dir=recovered_training, training_source_root=ROOT, output_dir=tmp_path / "rejected-restart")
    assert not (tmp_path / "rejected-restart").exists()


def test_recovery_receipts_relocate_with_pinned_adjacent_driver(recovered_training, tmp_path):
    import shutil

    summary = json.loads((recovered_training / "summary.json").read_text())
    tag = summary["recovery"]["recovery_id"]
    driver = Path(summary["recovery"]["driver_path"])
    shutil.copyfile(driver, recovered_training / f"recovery_attempts/{tag}/recovery_driver.py")
    moved = tmp_path / "fresh-checkout/results/gradient_development"
    moved.parent.mkdir(parents=True)
    recovered_training.rename(moved)
    assert not driver.exists()
    frozen = confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
        training_dir=moved, training_source_root=ROOT, output_dir=tmp_path / "portable-freeze")
    assert frozen["training_admission"]["mode"] == "explicit-audited-ENOSPC-restart"
    confirmation.validate_selection_freeze(tmp_path / "portable-freeze")
