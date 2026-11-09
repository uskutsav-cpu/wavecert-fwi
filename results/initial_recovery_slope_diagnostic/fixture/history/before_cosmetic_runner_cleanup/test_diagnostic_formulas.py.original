"""Focused manufactured-array/mock guards; no original arrays, PDE or outcomes."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("manufactured_slope_diagnostic", HERE / "diagnose_initial_slopes.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def numeric(monkeypatch):
    monkeypatch.setattr(diagnostic, "np", np, raising=False)
    return diagnostic


def test_default_cli_only_reads_its_staged_source_and_protocol(capsys, monkeypatch):
    monkeypatch.delattr(diagnostic, "np", raising=False)
    original = diagnostic.file_sha
    seen = []

    def spy(path):
        seen.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(diagnostic, "file_sha", spy)
    assert diagnostic.main([]) == 0
    assert "source_only_no_project_reads_or_numeric_imports" in capsys.readouterr().out
    assert set(seen) == {(HERE / "diagnose_initial_slopes.py").resolve(), (HERE / "protocol.json").resolve()}
    assert "np" not in diagnostic.__dict__


def test_independent_hand_computed_half_squared_metric_and_derivative_numbers(numeric):
    # m=1, mt=4, v=1, vt=1/2, d=2 gives these exact rational values.
    model, truth, direction = np.array([1.0]), np.array([4.0]), np.array([2.0])
    values = numeric.metric_values(model, truth)
    slopes = numeric.metric_derivatives(model, truth, direction)
    assert values["half_relative_squared_m_error"] == 9 / 32
    assert values["half_relative_squared_velocity_error"] == 1 / 2
    assert slopes["half_relative_squared_m_error"]["analytic"] == -3 / 8
    assert slopes["half_relative_squared_velocity_error"]["analytic"] == -2
    assert values["relative_squared_slowness_L2"] == 3 / 4
    assert values["relative_physical_velocity_L2"] == 1


def test_fixed_two_scale_FD_validates_formula_keeps_positivity_and_does_not_clip(numeric):
    model, truth, direction = np.array([1.0]), np.array([4.0]), np.array([2.0])
    original_model = model.copy()
    checks = numeric.Checks()
    slopes = numeric.metric_derivatives(model, truth, direction)
    result = numeric.finite_difference_metrics(model, truth, direction, slopes, protocol(), checks, "hand_case")
    assert not checks.issues and result["status"] == "computed"
    assert result["actual_steps"] == [5e-5, 2.5e-5]
    assert result["actual_steps"][1] == result["actual_steps"][0] / 2
    for row in result["step_metadata"]:
        assert row["maximum_relative_coordinate_perturbation"] <= row["dimensionless_epsilon"] * (1 + 1e-12)
        assert row["minimum_positive_model_plus"] > 0 and row["minimum_positive_model_minus"] > 0
        assert row["box_violations_plus"] == 1
        assert row["box_violations_minus"] == 0
        assert row["no_projection_or_clipping"]
    np.testing.assert_array_equal(model, original_model)


def test_true_equals_initial_half_squared_metrics_have_zero_derivative(numeric):
    model = np.array([0.1, 0.2, 0.3])
    direction = np.array([0.3, -0.1, 0.2])
    values = numeric.metric_values(model, model.copy())
    derivatives = numeric.metric_derivatives(model, model.copy(), direction)
    assert values["half_relative_squared_m_error"] == values["half_relative_squared_velocity_error"] == 0
    assert all(row["analytic"] == 0 for row in derivatives.values())
    checks = numeric.Checks()
    result = numeric.finite_difference_metrics(model, model.copy(), direction, derivatives, protocol(), checks, "at_truth")
    assert result["status"] == "computed" and not checks.issues
    assert "quadratic metric" in result["metric_checks"]["half_relative_squared_m_error"][1]["two_scale_scope"]


def test_zero_direction_retains_unknown_cosine_and_no_empirical_FD_claim(numeric):
    model, truth = np.array([0.1, 0.2]), np.array([0.2, 0.3])
    checks = numeric.Checks()
    report = numeric.direction_report(model, truth, np.ones(2), np.zeros(2), protocol(), checks, "zero_direction")
    assert report["data_directional_derivative"]["literal_sign"] == "zero"
    assert report["native_m_cosine_direction_to_truth_correction"] is None
    assert report["finite_difference_metric_validation"]["status"] == "zero_direction_trivial"
    assert not report["finite_difference_metric_validation"]["actual_steps"]
    assert not checks.issues


def test_unrepresentable_positive_domain_step_is_flagged_unknown_not_zero(numeric):
    model = np.array([np.nextafter(0.0, 1.0)])
    derivatives = {name: {"analytic": 0.0, "uncancelled_derivative_scale": 1.0,
                          "metric_evaluation_scale": 1.0}
                   for name in ("half_relative_squared_m_error", "half_relative_squared_velocity_error")}
    checks = numeric.Checks()
    result = numeric.finite_difference_metrics(model, model.copy(), np.ones(1), derivatives,
                                               protocol(), checks, "underflow_control")
    assert result["status"] == "step_not_representable"
    assert result["unavailable_FD_information_is_unknown_not_zero"]
    assert checks.issues
    assert "metric_checks" not in result


def test_literal_box_violations_stay_violations_inside_near_FP_band(numeric):
    bounds = protocol()["original_squared_slowness_box"]
    lower, upper = bounds
    values = np.array([lower, upper, np.nextafter(lower, 0.0), np.nextafter(upper, np.inf),
                       lower - 0.001, upper + 0.001])
    original = values.copy()
    result = numeric.feasibility_record(values, protocol())
    assert result["finite_positive"] and not result["literal_box_feasible"]
    assert result["literal_below_lower_cells"] == result["literal_above_upper_cells"] == 2
    assert result["exact_lower_bound_cells"] == result["exact_upper_bound_cells"] == 1
    assert result["near_lower_FP_band_cells"] == result["near_upper_FP_band_cells"] == 2
    assert result["literal_below_but_inside_lower_FP_band_cells"] == 1
    assert result["literal_above_but_inside_upper_FP_band_cells"] == 1
    np.testing.assert_array_equal(values, original)


def test_feasibility_nonfinite_and_nonpositive_controls_are_explicit(numeric):
    result = numeric.feasibility_record(np.array([0.1, 0.0, -0.1, np.nan, np.inf]), protocol())
    assert result["finite_cells"] == 3 and result["nonfinite_cells"] == 2
    assert result["nonpositive_finite_cells"] == 2
    assert not result["finite_positive"] and not result["literal_box_feasible"]
    assert result["finite_minimum"] == -0.1 and result["finite_maximum"] == 0.1


def test_manufactured_local_opposition_is_only_local_and_signs_are_separate(numeric):
    model, truth = np.array([0.1, 0.2]), np.array([0.3, 0.2])
    checks = numeric.Checks()
    result = numeric.direction_report(model, truth, np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                                      protocol(), checks, "invented_gradient")
    assert result["data_directional_derivative"]["value"] == -1
    assert result["native_m_cosine_direction_to_truth_correction"]["value"] == -1
    assert all(result["literal_local_opposition"].values())
    assert all(result["numerically_classified_local_opposition"].values())
    assert "no finite step/causal proof" in result["scope"]
    assert not checks.issues
    tiny = numeric.sign_record(1e-14, 1.0, protocol())
    assert tiny["literal_sign"] == "positive" and tiny["numerical_sign"] == "near_zero"


@pytest.mark.parametrize("exception_class", [KeyboardInterrupt, SystemExit])
def test_handled_interrupt_preserves_finished_case_prefix_and_other_slots(tmp_path, monkeypatch, exception_class):
    # Deliberately mock all manifests/provenance and supply generated 2-cell vectors.
    # This is an output-state guard control, not a real upstream artifact audit.
    repo, witness, benchmark = [tmp_path / name for name in ("invented_repo", "invented_witness", "invented_benchmark")]
    for path in (repo, witness, benchmark):
        path.mkdir()
    original = {"identities": [{"case": i} for i in range(8)], "inputs_sha256": {}}
    raw_original = json.dumps(original).encode()
    (benchmark / "historical_candidate_protocol.json").write_bytes(raw_original)
    (witness / "original_protocol.json").write_bytes(raw_original)
    monkeypatch.setattr(diagnostic, "ORIGINAL_PROTOCOL_SHA", hashlib.sha256(raw_original).hexdigest())
    source = {"sha256": diagnostic.PRODUCTION_SHA, "files": 161}
    benchmark_source = {"candidate": {"repository_source_sha256": diagnostic.PRODUCTION_SHA}}
    witness_status = {"status": "complete_initial_direction_verification", "verification_result": "clear",
        "source_before": source, "source_after": source, "cases": [{"case_index": i} for i in range(8)]}
    benchmark_status = {"status": "complete_engineering_benchmark", "source_before": benchmark_source,
                        "source_after": benchmark_source, "runs": [{} for _ in range(360)]}

    def mock_manifest(path, expected_sha, key):
        return (witness_status if path == witness else benchmark_status), {"mock_metadata": True}

    monkeypatch.setattr(diagnostic, "validate_manifest", mock_manifest)
    monkeypatch.setattr(diagnostic, "source_receipt", lambda _: source.copy())

    def invented_vectors(path, row, *args):
        if row["case_index"] == 1:
            raise exception_class("manufactured interruption after finished first case")
        return {"initial_model": np.array([0.1, 0.2]),
                "independent_truth_model": np.array([0.2, 0.3]),
                "independent_true_gradient": np.zeros(2), "approximate_gradient": None, "direction": None}, {"mock": True}

    monkeypatch.setattr(diagnostic, "load_initial_case", invented_vectors)
    output = tmp_path / "new_mock_diagnostic"
    with pytest.raises(exception_class):
        diagnostic.run(repo, witness, benchmark, output, HERE / "protocol.json",
                       checksum(HERE / "diagnose_initial_slopes.py"), checksum(HERE / "protocol.json"))
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed_or_interrupted"
    assert status["case_slots"][0]["status"] == "complete"
    assert status["case_slots"][1]["status"] == "failed_or_interrupted"
    assert all(row["status"] == "unattempted" for row in status["case_slots"][2:])
    first = json.loads((output / "case_00_slopes.json").read_text())
    assert first["directions"]["exact_normalized_negative_data_gradient"]["available"] is False
    assert "no normalized negative direction" in first["directions"]["exact_normalized_negative_data_gradient"]["reason"]
    assert status["case_slots"][0]["artifact_sha256"] == checksum(output / "case_00_slopes.json")
    assert not (output / "complete_receipt.json").exists()


def test_machine_epsilon_band_multiplier_is_fixed_not_case_selected():
    assert protocol()["bound_FP_band_multiplier"] == 64
    assert protocol()["finite_difference_dimensionless_epsilons"] == [1e-4, 5e-5]
    assert sys.float_info.epsilon > 0
