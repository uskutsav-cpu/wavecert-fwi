import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_gradient_confirmation import completed_training, training_provenance  # noqa: F401

from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.experiments import pod_confirmation as pod
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.provenance import sha256_file
from wavecert.surrogates.pod import PODGalerkinSurrogate

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/gradient_confirmation_plan.json"
ADDENDUM = ROOT / "configs/pod_confirmation_addendum.json"


@pytest.fixture
def teacher_receipt_files(tmp_path, monkeypatch):
    """Mock binary teacher I/O; no large archives or held-out arrays needed."""
    teacher = tmp_path / "teacher"
    expected = json.loads((ROOT / "results/pod_development/input_provenance.json").read_text())["verified_file_sha256"]
    standins = {}
    for name, checksum in expected.items():
        path = teacher / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"binary teacher archive stand-in:{name}".encode()
        path.write_bytes(data)
        standins[path.resolve()] = (data, checksum)
    original = pod.sha256_file
    def mocked_hash(path):
        item = standins.get(Path(path).resolve())
        return item[1] if item is not None and Path(path).read_bytes() == item[0] else original(path)
    monkeypatch.setattr(pod, "sha256_file", mocked_hash)
    return teacher


def test_pod_addendum_audits_all_eight_and_keeps_original_neural_plan(teacher_receipt_files):
    addendum, plan, summary = pod.validate_pod_addendum(ADDENDUM, ROOT, teacher_study=teacher_receipt_files)
    assert len(summary["candidates"]) == 8
    assert addendum["selected_pod"]["family"] == "state+tangent"
    assert addendum["selected_pod"]["rank"] == 64
    assert [row["label"] for row in plan["legacy_models"]] == ["legacy_forward", "legacy_jvp"]
    assert addendum["source_calibration"]["pod_calibration_performed"] is False
    assert sha256_file(PLAN) == confirmation.FROZEN_PLAN_SHA256


def test_changed_pod_addendum_is_rejected_before_teacher_load(tmp_path):
    value = json.loads(ADDENDUM.read_text())
    value["selected_pod"]["rank"] = 32
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="addendum bytes"):
        pod.validate_pod_addendum(path, ROOT, teacher_study=tmp_path / "absent")


def test_pod_addendum_rejects_teacher_tamper_without_loading_outcomes(tmp_path):
    (tmp_path / "protocol.json").write_bytes(b"changed original teacher protocol")
    with pytest.raises(ValueError, match="teacher input"):
        pod.validate_pod_addendum(ADDENDUM, ROOT, teacher_study=tmp_path)


def test_pod_prediction_cannot_precede_all_three_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(PODGalerkinSurrogate, "load", lambda *a, **k: pytest.fail("POD loaded before frozen gates"))
    with pytest.raises(FileNotFoundError):
        pod.run_joint_pod_confirmation(addendum_path=ADDENDUM, plan_path=PLAN, repository_root=ROOT,
            selection_dir=tmp_path / "absent", calibration_dir=tmp_path / "absent", pod_dir=tmp_path / "absent",
            data_dir=tmp_path, output_dir=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_pod_freeze_is_separate_immutable_and_keeps_all_basis_identities(completed_training, tmp_path, teacher_receipt_files):  # noqa: F811
    selection = tmp_path / "neural-freeze"
    confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
        training_dir=completed_training, training_source_root=ROOT, output_dir=selection)
    paths = [ADDENDUM, PLAN, *list((ROOT / "results/pod_development").rglob("*.json")),
             *list((ROOT / "results/pod_development").rglob("*.npz"))]
    original = {p: sha256_file(p) for p in paths}
    output = tmp_path / "pod-freeze"
    frozen = pod.freeze_pod_comparator(addendum_path=ADDENDUM, repository_root=ROOT,
        teacher_study=teacher_receipt_files, selection_dir=selection, output_dir=output)
    assert len(frozen["all_eight_candidates"]) == 8
    assert frozen["fresh_predictions_performed_by_this_stage"] is False
    assert frozen["source_calibrated_pod_gate_evaluated"] is False
    pod.validate_pod_freeze(output, selection)
    assert all(sha256_file(path) == checksum for path, checksum in original.items())
    (output / frozen["basis_path"]).write_bytes(b"substituted basis")
    with pytest.raises(ValueError, match="input changed"):
        pod.validate_pod_freeze(output, selection)


def test_pod_actual_sum_gradient_matches_independent_objective_finite_difference():
    physics = Helmholtz2D(shape=(4, 4), spacing=.2, damping_width=1, damping_strength=2.)
    basis, _ = np.linalg.qr(np.random.default_rng(40).normal(size=(16, 7)))
    surrogate = PODGalerkinSurrogate(physics, basis)
    rng = np.random.default_rng(7)
    model = np.linspace(.2, .45, 16)
    direction = rng.normal(size=16)
    direction /= np.linalg.norm(direction)
    receivers = (1, 2, 2, 8, 12)
    survey = [SurveyBlock(source, frequency, rng.normal(size=5) + 1j * rng.normal(size=5), str(i))
              for i, (source, frequency) in enumerate(((1, 3.), (7, 3.), (1, 4.), (7, 4.)))]
    _, gradient = pod.pod_survey_objective_and_gradient(surrogate, physics, model, survey, receivers)
    def value(m):
        return sum(surrogate.objective(m, physics.source(block.source_index), block.observed,
                                     receivers, block.frequency_hz) for block in survey)
    step = 1e-6
    numerical = (value(model + step * direction) - value(model - step * direction)) / (2 * step)
    assert np.dot(gradient, direction) == pytest.approx(numerical, rel=2e-6, abs=1e-8)


def _pod_rows():
    plan = json.loads(PLAN.read_text())
    blocks, models, audits = [], [], {}
    for family in plan["families"]:
        name = family["name"]
        audits[name] = {"population_unique_raw_hashes": 400}
        for index in family["confirmation_selected_indices"]:
            models.append({"family": name, "checkpoint": pod.LABEL, "original_index": index,
                "aggregate_gradient_cosine": .8, "aggregate_neural_true_descent": True})
            for source, frequency in ((1, 3.), (2, 3.), (1, 4.), (2, 4.)):
                blocks.append({"family": name, "checkpoint": pod.LABEL, "original_index": index,
                    "source_index": source, "frequency_hz": frequency, "deterministic_accepts": True,
                    "deterministic_false_acceptance": False, "deterministic_covers": True,
                    "source_calibrated_covers": None, "source_calibrated_accepts": None,
                    "source_calibrated_false_acceptance": None})
    return blocks, models, plan, audits


def test_pod_summary_clusters_four_blocks_uses_N400_and_has_no_fake_calibration():
    blocks, models, plan, audits = _pod_rows()
    blocks[0]["deterministic_covers"] = False
    result = pod.summarize_pod_confirmation(blocks, models, plan, audits)
    family = result[plan["families"][0]["name"]]
    assert family["analysis_population_size"] == 400
    event = family["fixed_complement_hypergeometric_bounds"]["model_deterministic_coverage_failure"]
    assert event["observed_failures"] == 1 and event["sample_size"] == 100
    assert "not evaluated" in family["source_calibrated_gate"]
    assert family["primary_mean_actual_aggregate_gradient_cosine"] == pytest.approx(.8)
    assert family["conditional_model_bootstrap_aggregate_cosine"]["interval"] == pytest.approx([.8, .8])
    bad = copy.deepcopy(blocks)
    bad[0]["source_calibrated_covers"] = True
    with pytest.raises(ValueError, match="no frozen source-calibrated"):
        pod.summarize_pod_confirmation(bad, models, plan, audits)
    with pytest.raises(ValueError, match="every selected"):
        pod.summarize_pod_confirmation(blocks[:-1], models, plan, audits)
