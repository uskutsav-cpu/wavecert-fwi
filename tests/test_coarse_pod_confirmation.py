import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_gradient_confirmation import completed_training, training_provenance  # noqa: F401
from test_pod_confirmation import teacher_receipt_files  # noqa: F401

from wavecert.experiments import coarse_pod_confirmation as coarse
from wavecert.experiments import gradient_confirmation as confirmation
from wavecert.experiments import pod_confirmation as pod
from wavecert.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/gradient_confirmation_plan.json"
POD_PLAN = ROOT / "configs/pod_confirmation_addendum.json"
ADDENDUM = ROOT / "configs/coarse_corrected_pod_confirmation_addendum.json"


def test_fifth_addendum_preserves_both_previous_plans_and_declares_later_selection():
    value = coarse.validate_coarse_addendum(ADDENDUM, ROOT)
    assert len(value["model_labels"]) == 5
    assert "AFTER" in value["choice_timing"]
    assert value["fixed_method"]["coarse_stability_rank"] == 96
    assert sha256_file(PLAN) == confirmation.FROZEN_PLAN_SHA256
    assert sha256_file(POD_PLAN) == pod.ADDENDUM_SHA256


def test_fifth_prediction_and_setup_cannot_precede_prior_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(coarse, "_make_components", lambda *a, **k: pytest.fail("coarse model/setup before gates"))
    with pytest.raises(FileNotFoundError):
        coarse.freeze_coarse_pod_comparator(addendum_path=ADDENDUM, repository_root=ROOT,
            selection_dir=tmp_path / "missing", calibration_dir=tmp_path / "missing", pod_dir=tmp_path / "missing",
            output_dir=tmp_path / "forbidden-freeze")
    with pytest.raises(FileNotFoundError):
        coarse.run_five_model_confirmation(addendum_path=ADDENDUM, pod_addendum_path=POD_PLAN, plan_path=PLAN,
            repository_root=ROOT, selection_dir=tmp_path / "missing", calibration_dir=tmp_path / "missing",
            pod_dir=tmp_path / "missing", coarse_dir=tmp_path / "missing", data_dir=tmp_path,
            output_dir=tmp_path / "forbidden-outcomes")
    assert not (tmp_path / "forbidden-freeze").exists()
    assert not (tmp_path / "forbidden-outcomes").exists()


def _result(unavailable=False, wrong_exact_sign=False):
    blocks, reference = [], {}
    for source, frequency in ((1, 3.), (2, 3.), (1, 4.), (2, 4.)):
        available = not (unavailable and source == 1 and frequency == 3.)
        blocks.append(SimpleNamespace(source_index=source, frequency_hz=frequency,
            physical_gradient=np.array([1., 0.]), approximate_objective=1., approximate_directional_derivative=-1.,
            available=available, directional_error_bound=.1 if available else None,
            correction_applied=available, correction_skipped=not available, reason=None if available else "gamma<=0"))
        reference[source, frequency] = (2., np.array([-1., 0.]) if wrong_exact_sign else np.array([1., 0.]))
    result = SimpleNamespace(approximate_gradient=np.array([4., 0.]), direction=np.array([-1., 0.]),
        approximate_objective=4., approximate_directional_derivative=-4., block_certificates=blocks,
        available=not unavailable, directional_error_bound=.4 if not unavailable else None,
        certified_upper_derivative=-3.6 if not unavailable else None, certified_descent=not unavailable,
        exact_fallback_required=unavailable)
    return result, reference


def test_physical_aggregate_uses_shared_direction_and_preserves_failed_certificate_outcomes():
    result, reference = _result(wrong_exact_sign=True)
    blocks, model, exact, proposal = coarse.coarse_result_records(result, reference, identity={"checkpoint": coarse.LABEL})
    assert len(blocks) == 4
    assert model["aggregate_deterministic_false_acceptance"] is True
    assert model["aggregate_deterministic_covers"] is False
    assert model["aggregate_approximate_true_descent"] is False
    assert model["aggregate_gradient_cosine"] == pytest.approx(-1.)
    assert np.all(exact == -proposal)


def test_unavailable_certificate_keeps_paid_proposal_and_strict_null_abstention():
    result, reference = _result(unavailable=True)
    blocks, model, _, proposal = coarse.coarse_result_records(result, reference, identity={"checkpoint": coarse.LABEL})
    assert np.linalg.norm(proposal) == 4
    assert model["certificate_available"] is False and model["certified_descent"] is False
    assert model["exact_fallback_required"] is True
    assert model["aggregate_deterministic_bound"] is None
    assert model["aggregate_certified_upper_derivative"] is None
    assert model["aggregate_deterministic_covers"] is None
    assert blocks[0]["correction_skipped"] is True and blocks[0]["deterministic_bound"] is None
    json.dumps({"model": model, "blocks": blocks}, allow_nan=False)


@pytest.mark.parametrize("change", ["direction", "block_direction", "missing_block", "nan_proposal", "false_availability", "clipped_unavailable_bound", "missing_fallback"])
def test_physical_aggregate_rejects_material_inconsistency(change):
    result, reference = _result(unavailable=change in {"false_availability", "clipped_unavailable_bound", "missing_fallback"})
    if change == "direction":
        result.direction = np.array([0., -1.])
    elif change == "block_direction":
        result.block_certificates[0].approximate_directional_derivative = 1.
    elif change == "missing_block":
        result.block_certificates = result.block_certificates[:-1]
    elif change == "nan_proposal":
        result.approximate_gradient[0] = np.nan
    elif change == "false_availability":
        result.available = True
    elif change == "clipped_unavailable_bound":
        result.directional_error_bound = 0.
    else:
        result.exact_fallback_required = False
    with pytest.raises(ValueError):
        coarse.coarse_result_records(result, reference, identity={"checkpoint": coarse.LABEL})


def test_fifth_summary_does_not_remove_unavailable_models_from_denominator():
    plan = json.loads(PLAN.read_text())
    blocks, models, audits = [], [], {}
    for family in plan["families"]:
        audits[family["name"]] = {"population_unique_raw_hashes": 400}
        for order, index in enumerate(family["confirmation_selected_indices"]):
            result, reference = _result(unavailable=order == 0)
            identity = {"family": family["name"], "checkpoint": coarse.LABEL, "original_index": index}
            rows, model, _, _ = coarse.coarse_result_records(result, reference, identity=identity)
            blocks.extend(rows)
            models.append(model)
    summary = coarse.summarize_coarse_confirmation(blocks, models, plan, audits)
    for family in summary.values():
        assert family["sampled_original_index_slots"] == 100
        assert family["certificate_available_models"] == 99
        assert family["exact_fallback_required_models"] == 1
        event = family["fixed_complement_hypergeometric_bounds"]["certificate_unavailable"]
        assert event["observed_failures"] == 1 and event["sample_size"] == 100 and event["population_size"] == 400
    bad = copy.deepcopy(models)
    bad.pop()
    with pytest.raises(ValueError):
        coarse.summarize_coarse_confirmation(blocks, bad, plan, audits)


@pytest.mark.parametrize("frequencies", [(1., 1.3), (1., 20.)])
def test_public_coarse_api_matches_adapter_on_tiny_synthetic_available_and_unavailable_cases(frequencies):
    foundation = pytest.importorskip("wavecert.physics.coarse_stability")
    certificates = pytest.importorskip("wavecert.certificates.coarse_adjoint")
    from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
    from wavecert.surrogates.pod import PODGalerkinSurrogate

    physics = Helmholtz2D(shape=(4, 4), spacing=.2, damping_width=1, damping_strength=2.)
    basis, _ = np.linalg.qr(np.random.default_rng(40).normal(size=(16, 7)))
    surrogate = PODGalerkinSurrogate(physics, basis)
    stability = foundation.FastCoarseStability(physics, rank=8)
    adapter = certificates.PODCoarseAdjointCertifier(physics, surrogate, stability)
    model = np.linspace(.2, .45, 16)
    receivers = (1, 2, 2, 8, 12)
    rng = np.random.default_rng(6)
    survey, reference = [], {}
    for frequency in frequencies:
        for source in (1, 7):
            observed = rng.normal(size=5) + 1j * rng.normal(size=5)
            survey.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
            value, gradient, _, _ = physics.objective_and_gradient(model, physics.source(source), observed, receivers, frequency)
            reference[source, frequency] = (value, gradient)
    result = adapter.certify_survey(model, survey, receivers, method="coarse_corrected_oriented")
    coarse.validate_result_context(result.to_dict(), model, stability.operator_signature, frequencies)
    blocks, summary, exact, proposed = coarse.coarse_result_records(result, reference, identity={"checkpoint": coarse.LABEL})
    assert len(blocks) == 4 and exact.shape == proposed.shape == (16,)
    assert np.all(np.isfinite(proposed))
    assert all(result.costs["reference_solver"].get(key, 0) == 0 for key in ("factorizations", "forward_rhs", "adjoint_rhs", "stability_rhs"))
    if frequencies[-1] == 20.:
        assert not summary["certificate_available"]
        assert summary["aggregate_deterministic_bound"] is None
        assert summary["exact_fallback_required"]
    json.dumps(result.to_dict(), allow_nan=False)
    changed = copy.deepcopy(result.to_dict())
    changed["model_sha256"] = "different physical model"
    with pytest.raises(ValueError, match="context differs"):
        coarse.validate_result_context(changed, model, stability.operator_signature, frequencies)


@pytest.fixture
def integrated_freeze_inputs(request, tmp_path, monkeypatch):
    """Run only after public modules are present in this same checkout."""
    foundation = pytest.importorskip("wavecert.physics.coarse_stability")
    pytest.importorskip("wavecert.certificates.coarse_adjoint")
    if Path(foundation.__file__).resolve() != ROOT / "src/wavecert/physics/coarse_stability.py":
        pytest.skip("positive freeze integration requires the reviewed modules merged into this checkout")
    training = request.getfixturevalue("completed_training")
    teacher = request.getfixturevalue("teacher_receipt_files")
    from wavecert.experiments import real_wavecert

    selection, calibration, pod_dir = tmp_path / "B", tmp_path / "C", tmp_path / "POD"
    confirmation.freeze_selected_checkpoint(plan_path=PLAN, repository_root=ROOT,
        training_dir=training, training_source_root=ROOT, output_dir=selection)
    def fake_calibration(**kwargs):
        report = {"calibration_parent_models": 12, "alpha": .1, "conformal_scale": .5,
            "checkpoint_sha256": sha256_file(kwargs["checkpoint_path"]),
            "dataset_provenance": {"dataset_manifest_sha256": json.loads(PLAN.read_text())["selected_calibration"]["dataset_manifest_sha256"]},
            "id": {"n_parent_models": 8}, "ood": {"n_parent_models": 1}}
        path = Path(kwargs["output_dir"]) / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
        return report
    monkeypatch.setattr(real_wavecert, "run_real_wavecert_study", fake_calibration)
    confirmation.run_selected_calibration(freeze_dir=selection, repository_root=ROOT,
        dataset_dir=ROOT / "data/reference/subsurfacegen_parent_split", output_dir=calibration)
    pod.freeze_pod_comparator(addendum_path=POD_PLAN, repository_root=ROOT, teacher_study=teacher,
        selection_dir=selection, output_dir=pod_dir)
    return selection, calibration, pod_dir


def test_fifth_operator_only_freeze_binds_prior_stages_without_outcome_predictions(integrated_freeze_inputs, tmp_path):
    selection, calibration, pod_dir = integrated_freeze_inputs
    output = tmp_path / "coarse-freeze"
    frozen = coarse.freeze_coarse_pod_comparator(addendum_path=ADDENDUM, repository_root=ROOT,
        selection_dir=selection, calibration_dir=calibration, pod_dir=pod_dir, output_dir=output)
    assert frozen["operator_signature"]["rank"] == 96 and frozen["operator_signature"]["n"] == 576
    assert frozen["fresh_predictions_performed_by_this_stage"] is False
    assert frozen["source_calibrated_gate_evaluated"] is False
    coarse.validate_coarse_pod_freeze(output, selection, calibration, pod_dir, ROOT)
    (output / "pod_basis.npz").write_bytes(b"substituted basis")
    with pytest.raises(ValueError, match="input changed"):
        coarse.validate_coarse_pod_freeze(output, selection, calibration, pod_dir, ROOT)


def test_numeric_aggregate_abstention_cannot_be_marked_a_complete_model_record():
    result, reference = _result()
    result.available, result.certified_descent, result.exact_fallback_required = False, False, True
    result.directional_error_bound = result.certified_upper_derivative = None
    with pytest.raises(ValueError, match="numeric aggregate abstention"):
        coarse.coarse_result_records(result, reference, identity={"checkpoint": coarse.LABEL})
    result.approximate_directional_derivative = None
    with pytest.raises(ValueError, match="numeric aggregate abstention"):
        coarse.coarse_result_records(result, reference, identity={"checkpoint": coarse.LABEL})


def test_synthetic_staging_uses_only_current_head_when_old_git_history_is_unavailable(request, tmp_path, monkeypatch):
    import subprocess

    from test_gradient_confirmation import _current_head_source_provenance

    original = subprocess.check_output
    git_calls = []
    def shallow_git(arguments, *args, **kwargs):
        if arguments and arguments[0] == "git":
            git_calls.append(tuple(arguments))
            assert not any("be6e51e" in str(value) or "6470e4d" in str(value) for value in arguments), "synthetic fixtures requested historical Git objects"
            assert arguments[1] not in {"fetch", "pull"}, "unit fixtures must not fetch missing history"
        return original(arguments, *args, **kwargs)
    monkeypatch.setattr(subprocess, "check_output", shallow_git)
    current = _current_head_source_provenance()
    selection, calibration, pod_dir = request.getfixturevalue("integrated_freeze_inputs")
    output = tmp_path / "shallow-compatible-fifth"
    frozen = coarse.freeze_coarse_pod_comparator(addendum_path=ADDENDUM, repository_root=ROOT,
        selection_dir=selection, calibration_dir=calibration, pod_dir=pod_dir, output_dir=output)
    coarse.validate_coarse_pod_freeze(output, selection, calibration, pod_dir, ROOT)
    plan = json.loads((selection / "plan.json").read_text())
    training = json.loads((selection / "training/summary.json").read_text())
    pod_summary = json.loads((pod_dir / "development/summary.json").read_text())
    assert plan["synthetic_test_fixture"] and training["synthetic_test_fixture"] and pod_summary["synthetic_test_fixture"]
    assert plan["development_training"]["source_commit"] == training["provenance"]["git_revision"] == pod_summary["provenance"]["git_revision"] == current["git_revision"]
    assert training["provenance"]["source_files"] == pod_summary["provenance"]["source_files"] == current["source_files"]
    assert ("git", "rev-parse", "HEAD") in git_calls
    assert any(call[1] == "show" and call[2].startswith(current["git_revision"] + ":") for call in git_calls)
    assert len(frozen["files"]) == 2
