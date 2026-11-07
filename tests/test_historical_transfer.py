import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_coarse_pod_confirmation import integrated_freeze_inputs  # noqa: F401
from test_gradient_confirmation import completed_training, training_provenance  # noqa: F401
from test_pod_confirmation import teacher_receipt_files  # noqa: F401

from wavecert.certificates.coarse_adjoint import PODCoarseAdjointCertifier
from wavecert.experiments import coarse_pod_confirmation as coarse
from wavecert.experiments import historical_transfer as historical
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.provenance import environment_record, sha256_file
from wavecert.surrogates.pod import PODGalerkinSurrogate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/historical_transfer_protocol.json"
DATA = ROOT / "data/reference/subsurfacegen_parent_split"


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def test_protocol_counts_every_cached_slice_without_opening_test_arrays(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("test NPZ opened during metadata-only protocol check"))
    protocol, metadata, audit = historical.validate_historical_protocol(PROTOCOL, ROOT)
    assert audit["parent_isolation_passed"]
    assert [len(metadata[role]) for role in ("test_id", "test_ood")] == [384, 96]
    assert [protocol["dataset"]["roles"][role]["n_slices"] for role in ("test_id", "test_ood")] == [16, 4]
    assert len(protocol["dataset"]["roles"]["test_ood"]["parents"]["penobscot_000"]) == 4


def test_project_array_loading_and_models_require_historical_freeze_first(tmp_path, monkeypatch):
    monkeypatch.setattr(historical, "load_historical_arrays", lambda *a, **k: pytest.fail("project arrays opened before freeze"))
    monkeypatch.setattr(historical.TrainedFNOWavefieldSurrogate, "load", lambda *a, **k: pytest.fail("model opened before freeze"))
    with pytest.raises(FileNotFoundError):
        historical.run_historical_transfer(freeze_dir=tmp_path / "absent", repository_root=ROOT,
            dataset_dir=DATA, output_dir=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_changed_historical_protocol_cannot_select_an_ood_subset(tmp_path):
    changed = json.loads(PROTOCOL.read_text())
    changed["dataset"]["roles"]["test_ood"]["parents"]["penobscot_000"] = changed["dataset"]["roles"]["test_ood"]["parents"]["penobscot_000"][:2]
    path = tmp_path / "changed.json"
    _save(path, changed)
    with pytest.raises(ValueError, match="protocol changed"):
        historical.validate_historical_protocol(path, ROOT)


def test_phase_admission_rejects_old_edits_and_unapproved_imported_additions(monkeypatch):
    current = environment_record(ROOT)
    previous = copy.deepcopy(current)
    previous["source_files"] = {name: value for name, value in current["source_files"].items() if name not in historical.PHASE_FILES}
    admission = historical.phase_source_admission(previous, current, ROOT)
    assert set(admission["added_phase_files"]) == historical.PHASE_FILES
    changed = copy.deepcopy(current)
    old_name = next(iter(previous["source_files"]))
    changed["source_files"][old_name] = "changed existing source"
    with pytest.raises(ValueError, match="previously frozen source"):
        historical.phase_source_admission(previous, changed, ROOT)
    changed = copy.deepcopy(current)
    name = "scripts/unapproved_historical_fixture.py"
    changed["source_files"][name] = "new source checksum"
    monkeypatch.setitem(sys.modules, "_unapproved_historical_fixture", SimpleNamespace(__file__=str(ROOT / name)))
    with pytest.raises(ValueError, match="unapproved new imported"):
        historical.phase_source_admission(previous, changed, ROOT)


def test_parent_actual_fno_module_must_match_frozen_checkout(monkeypatch):
    module = sys.modules[historical.TrainedFNOWavefieldSurrogate.__module__]
    monkeypatch.setattr(module, "__file__", "/foreign/wavecert/surrogates/fno.py")
    with pytest.raises(ValueError, match="another checkout"):
        historical._verify_historical_module_paths(ROOT)


def _survey(frequencies):
    physics = Helmholtz2D(shape=(4, 4), spacing=.2, damping_width=1, damping_strength=2.)
    basis, _ = np.linalg.qr(np.random.default_rng(40).normal(size=(16, 7)))
    pod = PODGalerkinSurrogate(physics, basis)
    model = np.linspace(.2, .45, 16)
    receivers = (1, 2, 2, 8, 12)
    rng = np.random.default_rng(8)
    survey, reference = [], {}
    for frequency in frequencies:
        for source in range(1, 7):
            observed = rng.normal(size=5) + 1j * rng.normal(size=5)
            survey.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
            value, gradient, _, _ = physics.objective_and_gradient(model, physics.source(source), observed, receivers, frequency)
            reference[source, frequency] = (value, gradient)
    return physics, pod, model, receivers, survey, reference


@pytest.mark.parametrize("frequencies", [(1., 1.1, 1.2, 1.3), (1., 1.1, 1.2, 20.)])
def test_actual24_block_corrected_api_preserves_complete_unavailable_outcomes(frequencies):
    physics, pod, model, receivers, survey, reference = _survey(frequencies)
    engine = FastCoarseStability(physics, rank=8)
    certifier = PODCoarseAdjointCertifier(physics, pod, engine)
    result = certifier.certify_survey(model, survey, receivers)
    rows, summary, exact, proposal, direction = historical.evaluate_corrected_result(result, model,
        engine.operator_signature, survey, reference, {"checkpoint": coarse.LABEL})
    assert len(rows) == 24 and summary["n_blocks"] == 24
    assert exact.shape == proposal.shape == direction.shape == (16,)
    assert summary["aggregate_approximate_directional_derivative"] == pytest.approx(np.dot(proposal, direction))
    if frequencies[-1] == 20.:
        assert summary["certificate_available"] is False
        assert summary["certified_descent"] is False and summary["exact_fallback_required"] is True
        assert summary["aggregate_deterministic_bound"] is None
        assert sum(row["correction_skipped"] for row in rows) == 6
    json.dumps({"rows": rows, "summary": summary}, allow_nan=False)


def test_historical_pod_sum_gradient_matches_independent24_block_objective_fd():
    physics, pod, model, receivers, survey, reference = _survey((1., 1.1, 1.2, 1.3))
    rows, summary, _, gradient, _ = historical.evaluate_uncertified_model(pod, "pod", physics,
        model, survey, receivers, reference, {"checkpoint": "classical_pod"})
    direction = np.random.default_rng(4).normal(size=16)
    direction /= np.linalg.norm(direction)
    def objective(m):
        return sum(pod.objective(m, physics.source(block.source_index), block.observed, receivers, block.frequency_hz) for block in survey)
    step = 1e-6
    numerical = (objective(model + step * direction) - objective(model - step * direction)) / (2 * step)
    assert np.dot(gradient, direction) == pytest.approx(numerical, rel=2e-6, abs=1e-8)
    assert summary["aggregate_approximate_objective"] == pytest.approx(objective(model))
    assert len(rows) == 24 and summary["certificate_available"] is None


def test_historical_fno_aggregate_uses_actual_shared_sum_and_counts_diagnostic_work():
    physics, _, model, receivers, survey, reference = _survey((1., 1.1, 1.2, 1.3))
    class Polynomial:
        calls, batches = 0, 0
        def objective_and_gradient(self, m, q, observed, indices, frequency):
            self.calls += 1
            weight = int(np.argmax(np.abs(q))) + frequency
            return weight * float(np.dot(m, m)), 2 * weight * m, None, None
        def survey_objective_and_gradient(self, m, blocks, indices):
            self.batches += 1
            weight = sum(block.source_index + block.frequency_hz for block in blocks)
            return weight * float(np.dot(m, m)), 2 * weight * m, None, None
    surrogate = Polynomial()
    _, summary, _, gradient, _ = historical.evaluate_uncertified_model(surrogate, "fno", physics,
        model, survey, receivers, reference, {"checkpoint": "development_winner"})
    total = sum(block.source_index + block.frequency_hz for block in survey)
    assert np.allclose(gradient, 2 * total * model)
    assert summary["aggregate_approximate_objective"] == pytest.approx(total * np.dot(model, model))
    assert surrogate.calls == 24 and surrogate.batches == 1
    costs = json.loads(summary["online_cost_json"])
    assert costs["total_neural_block_examples"] == 48


def test_all20_slices_are_clustered_into8_id_and1_ood_parent_without_iid_intervals():
    protocol = json.loads(PROTOCOL.read_text())
    blocks, models = [], []
    for role, cfg in protocol["dataset"]["roles"].items():
        for parent_order, (parent, slices) in enumerate(cfg["parents"].items()):
            for index, slice_id in enumerate(slices):
                cosine = (-1. if index == len(slices)-1 else 1.) if parent_order == 0 else 1.
                for label in historical.LABELS:
                    corrected = label == coarse.LABEL
                    available = index != 0
                    row = {"role": role, "model_id": parent, "slice_id": slice_id, "checkpoint": label,
                        "aggregate_gradient_cosine": cosine, "aggregate_approximate_true_descent": cosine > 0,
                        "certificate_available": available if corrected else None, "certified_descent": available if corrected else None,
                        "exact_fallback_required": not available if corrected else None,
                        "aggregate_deterministic_false_acceptance": False if corrected else None,
                        "aggregate_deterministic_covers": True if available and corrected else None}
                    models.append(row)
                    blocks.extend({"role": role, "model_id": parent, "slice_id": slice_id, "checkpoint": label,
                        "source_index": source, "frequency_hz": frequency} for source in protocol["geometry"]["source_indices"]
                        for frequency in protocol["geometry"]["frequencies_hz"])
    roles, parents = historical.summarize_historical_transfer(blocks, models, protocol)
    assert len(parents) == 45
    for label in historical.LABELS:
        assert roles["test_id"][label]["n_parent_groups"] == 8 and roles["test_id"][label]["n_slices"] == 16
        assert roles["test_ood"][label]["n_parent_groups"] == 1 and roles["test_ood"][label]["n_slices"] == 4
        assert roles["test_id"][label]["equal_parent_mean_actual_aggregate_gradient_cosine"] == pytest.approx(.875)
        assert roles["test_ood"][label]["equal_parent_mean_actual_aggregate_gradient_cosine"] == pytest.approx(.5)
        assert "confidence_interval" not in roles["test_id"][label]
    with pytest.raises(ValueError):
        historical.summarize_historical_transfer(blocks[:-1], models, protocol)


@pytest.fixture
def historical_prior_chain(request, tmp_path):
    pytest.importorskip("torch")
    selection, calibration, pod_dir = request.getfixturevalue("integrated_freeze_inputs")
    synthetic = request.getfixturevalue("synthetic_confirmation_artifacts")
    coarse_dir = tmp_path / "fifth"
    coarse.freeze_coarse_pod_comparator(addendum_path=synthetic["fifth_addendum"],
        repository_root=ROOT, selection_dir=selection, calibration_dir=calibration, pod_dir=pod_dir, output_dir=coarse_dir)
    def old_provenance(value):
        old = value["provenance"]
        old["source_files"] = {name: checksum for name, checksum in old["source_files"].items() if name not in historical.PHASE_FILES}
        digest = hashlib.sha256()
        for name, checksum in sorted(old["source_files"].items()):
            digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
        old["source_sha256"] = digest.hexdigest()
        return value
    B, C, P, E = selection / "freeze.json", calibration / "calibration_freeze.json", pod_dir / "pod_freeze.json", coarse_dir / "coarse_pod_freeze.json"
    _save(B, old_provenance(json.loads(B.read_text())))
    for path in (C, P):
        value = old_provenance(json.loads(path.read_text()))
        value["selection_freeze_sha256"] = sha256_file(B)
        _save(path, value)
    value = old_provenance(json.loads(E.read_text()))
    value.update(selection_freeze_sha256=sha256_file(B), calibration_freeze_sha256=sha256_file(C), pod_freeze_sha256=sha256_file(P))
    _save(E, value)
    return selection, calibration, pod_dir, coarse_dir


def test_new_historical_freeze_runs_old_guards_cleanly_and_never_opens_test_arrays(historical_prior_chain, tmp_path, monkeypatch):
    selection, calibration, pod_dir, coarse_dir = historical_prior_chain
    with pytest.raises(ValueError, match="newly added source file is imported"):
        coarse.validate_coarse_pod_freeze(coarse_dir, selection, calibration, pod_dir, ROOT)
    original = {p: sha256_file(p) for directory in historical_prior_chain for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("test arrays loaded during historical source/weight freeze"))
    output = tmp_path / "historical"
    frozen = historical.freeze_historical_transfer(protocol_path=PROTOCOL, repository_root=ROOT,
        selection_dir=selection, calibration_dir=calibration, pod_dir=pod_dir, coarse_dir=coarse_dir, output_dir=output)
    assert set(frozen["source_admission"]["added_phase_files"]) == historical.PHASE_FILES
    assert frozen["project_test_arrays_loaded_by_this_stage"] is False
    assert json.loads((output / "prior_guard_audit.json").read_text())["synthetic_test_fixture"] is True
    assert [model["label"] for model in frozen["models"]] == list(historical.LABELS)
    historical.validate_historical_freeze(output, ROOT)
    assert all(sha256_file(path) == checksum for path, checksum in original.items())
    frozen["models"][0]["kind"] = "pod"
    _save(output / "historical_freeze.json", frozen)
    with pytest.raises(ValueError, match="backend differs"):
        historical.validate_historical_freeze(output, ROOT)


def test_tampered_historical_npz_is_rejected_before_any_array_load(tmp_path, monkeypatch):
    protocol, metadata, _ = historical.validate_historical_protocol(PROTOCOL, ROOT)
    (tmp_path / "test_id.npz").write_bytes(b"changed cached test archive")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("tampered archive opened before hash verification"))
    with pytest.raises(ValueError, match="arrays changed before loading"):
        historical.load_historical_arrays(protocol, tmp_path, metadata)


def test_failed_neural_diagnostic_retains_attempted_and_completed_call_accounting():
    physics, _, model, receivers, survey, reference = _survey((1., 1.1, 1.2, 1.3))
    class Failed:
        calls = 0
        def objective_and_gradient(self, m, q, observed, indices, frequency):
            self.calls += 1
            if self.calls == 2:
                raise FloatingPointError("nonfinite neural response")
            return 1., np.ones_like(m), None, None
    accounting = {}
    with pytest.raises(FloatingPointError):
        historical.evaluate_uncertified_model(Failed(), "fno", physics, model, survey, receivers, reference,
            {"checkpoint": "development_winner"}, accounting=accounting)
    assert accounting == {"block_gradient_calls_attempted": 2, "block_gradient_calls_completed": 1}
