"""Portable regression of three known BP-crop traces and six derivatives.

V2 shares exact archived native-derived numeric INPUTS: probe velocity,
physical-m context and two directions. The independently implemented NumPy
solver completes all own solves before opening native SOLUTION OUTPUTS.
No installed Devito or new native run is needed. These reused crops share
one synthetic ancestor, not three sites or independent input generation.
"""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "bp_heterogeneous_verification"
FIXTURE = ARCHIVE / "numpy_replay_v2"
INPUTS = ARCHIVE / "canonical_inputs_v2_01"
RUNNER_SHA256 = "5c7f59a667a3cf8cdce56d65d8055792a3311c8ea20cab1f1470ddba12bb46e0"
PROTOCOL_SHA256 = "173873ed81968927c6a43edc3f37bdd7986aaa457582225e5df35d70948a938c"
NATIVE_COMPLETE_SHA256 = "ae2ca931876b6c1159af151394bfb0e9c0352ea0c384e879ac398e5bca019206"
TERMINAL_SHA256 = "75e54d29d1bf593471d9544363aa6c9d2b66ffa3db82d0aabe6e3d67c3fcd94d"
INPUT_MANIFEST_SHA256 = "2d878a87d947e8b4f1e9e69a5e487ab772e3de1156511a714a0a3b012fad3b35"
INPUT_SOURCE_SHA256 = "2a2e430a616d70a421b69fb5a76723844f10231b2ac5490852963917a51ea768"
EXPORT_PLAN_SHA256 = "bca57fc2fd89585b4b10176361f5ad9a0bd376bcb8d05c79b16937868eb39d48"
CORE_SHA256 = "b217dd6d2850f8c264f5910933cb1f4687bed8a7b3e780f7712c750d149328ce"
CASE_IDS = ["upper_left", "middle_center", "lower_right"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_array_difference(own, native):
    details = dict(own_shape=list(own.shape), native_shape=list(native.shape),
        own_dtype=str(own.dtype), native_dtype=str(native.dtype))
    if own.dtype != np.float64 or native.dtype != np.float64 or own.shape != native.shape:
        details["comparison_unavailable"] = "matching float64 shapes required"
        return details
    own = np.ascontiguousarray(own, dtype="<f8")
    native = np.ascontiguousarray(native, dtype="<f8")
    details.update(own_sha256=hashlib.sha256(own.tobytes()).hexdigest(),
        native_sha256=hashlib.sha256(native.tobytes()).hexdigest())
    if not (np.all(np.isfinite(own)) and np.all(np.isfinite(native))
            and np.all(own > 0) and np.all(native > 0)):
        details["comparison_unavailable"] = "finite positive models required for ULP ordering"
        return details
    own_bits, native_bits = own.view("<u8"), native.view("<u8")
    distances = np.maximum(own_bits, native_bits) - np.minimum(own_bits, native_bits)
    details.update(unequal_count=int(np.count_nonzero(own != native)),
        maximum_absolute_difference=float(np.max(np.abs(own - native))),
        maximum_ULP_distance=int(np.max(distances)))
    return details


def _saved_model_differences(row, output, native_run):
    assert _sha(native_run / "complete_receipt.json") == NATIVE_COMPLETE_SHA256
    receipt = json.loads((native_run / "complete_receipt.json").read_text())
    probe = next(call["result"] for call in receipt["calls"]
        if call["case_id"] == row["case_id"] and call["call_id"] == "probe_saved")
    own_path = output / row["artifact"]
    native_path = native_run / row["case_id"] / probe["artifact"]
    assert _sha(own_path) == row["artifact_sha256"]
    assert _sha(native_path) == probe["artifact_sha256"]
    with np.load(own_path, allow_pickle=False) as own, np.load(native_path, allow_pickle=False) as native:
        return {label: _model_array_difference(own[own_key], native[native_key])
            for label, own_key, native_key in (
                ("physical_velocity", "physical_probe_velocity", "actual_physical_velocity_kmps"),
                ("physical_m", "physical_probe_m", "actual_physical_m"))}


def _saved_input_differences(row, output):
    assert _sha(INPUTS / "manifest.json") == INPUT_MANIFEST_SHA256
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    canonical = next(case for case in manifest["cases"] if case["case"]["id"] == row["case_id"])
    own_path, input_path = output / row["artifact"], INPUTS / canonical["artifact"]
    assert _sha(own_path) == row["artifact_sha256"]
    assert _sha(input_path) == canonical["artifact_sha256"]
    result = {}
    with np.load(own_path, allow_pickle=False) as own, np.load(input_path, allow_pickle=False) as prescribed:
        for own_key, input_key in (("physical_probe_velocity", "physical_probe_velocity"),
                ("physical_probe_m", "direction_parameter_probe_m"),
                ("interior_direction", "interior_direction"), ("edge_direction", "edge_direction")):
            a, b = own[own_key], prescribed[input_key]
            details = dict(own_shape=list(a.shape), input_shape=list(b.shape),
                own_dtype=str(a.dtype), input_dtype=str(b.dtype))
            if a.shape == b.shape and a.dtype == np.float64 and b.dtype == np.float64:
                details.update(own_sha256=hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest(),
                    input_sha256=hashlib.sha256(np.ascontiguousarray(b, dtype="<f8").tobytes()).hexdigest(),
                    exact_value_identity=bool(np.array_equal(a, b)),
                    unequal_count=int(np.count_nonzero(a != b)))
                if np.isfinite(a).all() and np.isfinite(b).all():
                    details["maximum_absolute_difference"] = float(np.max(np.abs(a-b)))
            result[input_key] = details
    return result


def _replay_failure_diagnostics(completed, output, native_run=None):
    diagnostic = dict(check="bp_numpy_tangent_reproduction_failure",
        returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
        output=str(output))
    try:
        status = json.loads((output / "status.json").read_text())
        if not isinstance(status, dict):
            raise ValueError("saved status is not an object")
        diagnostic["status"] = {key: status.get(key) for key in (
            "status", "result", "error", "source_sha256", "protocol_sha256",
            "all3_own_complete_before_native_solution_outputs", "native_comparison_started",
            "native_solution_output_arrays_opened", "numerical_agreement_pass",
            "own_solves_planned", "directions_planned", "native_derived_numeric_inputs_used",
            "native_solution_outputs_used_during_own_solves", "canonical_input_manifest_sha256")}
        diagnostic["cases"] = []
        for row in status.get("cases", []):
            comparison = row.get("comparison") or row.get("comparison_prefix") or {}
            case_diagnostic = dict(
                case_id=row.get("case_id"), own_status=row.get("status"),
                error=row.get("error"), comparison_status=row.get("comparison_status"),
                comparison_error=row.get("comparison_error"),
                trace=comparison.get("trace"), trace_agreement=comparison.get("trace_agreement"),
                source_relative_L2=comparison.get("source_relative_L2"),
                padded_fold_relative_L2=comparison.get("padded_fold_relative_L2"),
                directions=[{key: direction.get(key) for key in (
                    "direction", "status", "informative", "agreement", "relative_error",
                    "absolute_error", "independent_tangent_DD", "native_folded_gradient_DD",
                    "direction_relative_L2")}
                    for direction in row.get("directions", [])])
            if native_run is not None and status.get("all3_own_complete_before_native_solution_outputs") is True:
                try:
                    case_diagnostic["saved_models"] = _saved_model_differences(row, output, native_run)
                    case_diagnostic["canonical_inputs"] = _saved_input_differences(row, output)
                except Exception as exc:
                    case_diagnostic["saved_model_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["cases"].append(case_diagnostic)
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        diagnostic["saved_status_error"] = f"{type(exc).__name__}: {exc}"
    return json.dumps(diagnostic, sort_keys=True)


def test_bp_numpy_traces_and_physical_m_tangents_without_devito(tmp_path, capsys):
    script, protocol_path = FIXTURE / "run_replay.py", FIXTURE / "protocol.json"
    native_run = ARCHIVE / "native_run_01"
    terminal_path = ARCHIVE / "native_terminal_observation_01.json"
    assert _sha(script) == RUNNER_SHA256
    assert _sha(protocol_path) == PROTOCOL_SHA256
    assert _sha(FIXTURE / "canonical_inputs.py") == INPUT_SOURCE_SHA256
    assert _sha(FIXTURE / "EXPORT_SOURCE_PLAN.json") == EXPORT_PLAN_SHA256
    assert _sha(FIXTURE / "numpy_bp_tangent.py") == CORE_SHA256
    assert _sha(INPUTS / "manifest.json") == INPUT_MANIFEST_SHA256
    assert _sha(native_run / "complete_receipt.json") == NATIVE_COMPLETE_SHA256
    assert _sha(terminal_path) == TERMINAL_SHA256
    terminal = json.loads(terminal_path.read_text())
    assert terminal["exit_code"] == 0 and terminal["complete_receipt_sha256"] == NATIVE_COMPLETE_SHA256
    assert terminal["call_status_counts"] == {"complete": 57}
    protocol = json.loads(protocol_path.read_text())
    assert [case["id"] for case in protocol["inputs"]] == CASE_IDS
    assert protocol["schema_version"] == 2
    assert protocol["canonical_input_source_sha256"] == INPUT_SOURCE_SHA256
    assert protocol["export_source_plan_sha256"] == EXPORT_PLAN_SHA256
    assert protocol["core_sha256"] == CORE_SHA256
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    assert [row["case"]["id"] for row in manifest["cases"]] == CASE_IDS
    assert manifest["plan_sha256"] == EXPORT_PLAN_SHA256
    assert manifest["exporter_sha256"] == INPUT_SOURCE_SHA256
    assert manifest["numeric_output_arrays_exported"] == 0
    assert manifest["native_complete_sha256"] == NATIVE_COMPLETE_SHA256
    assert protocol["criteria"] == dict(trace_relative_L2=1e-9, gradient_relative_error=1e-3,
        source_relative_L2=1e-14, direction_relative_L2=1e-14, fold_relative_L2=1e-13,
        near_zero_relative_normscale=1e-8)

    output = tmp_path / "bp-numpy-tangent-reproduction"
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-W", "error", str(script),
         "--protocol", str(protocol_path), "--execute-after-native-terminal",
         "--inputs", str(INPUTS), "--native-run", str(native_run),
         "--terminal-observation", str(terminal_path), "--output", str(output),
         "--expected-source-sha256", RUNNER_SHA256,
         "--expected-protocol-sha256", PROTOCOL_SHA256,
         "--expected-native-complete-sha256", NATIVE_COMPLETE_SHA256,
         "--expected-input-manifest-sha256", INPUT_MANIFEST_SHA256],
        check=False, capture_output=True, text=True, env=env, timeout=300,
    )
    assert completed.returncode == 0, _replay_failure_diagnostics(completed, output, native_run)
    result = json.loads((output / "complete_receipt.json").read_text())
    assert json.loads((output / "status.json").read_text()) == result
    assert result["status"] == "complete_comparison"
    assert result["result"] == "ALL_FIXED_NUMERICAL_CHECKS_AGREE"
    assert result["numerical_agreement_pass"] is True
    assert result["own_solves_planned"] == 3 and result["directions_planned"] == 6
    assert result["all3_own_complete_before_native_solution_outputs"] is True
    assert result["native_comparison_started"] is True and result["native_solution_output_arrays_opened"] is True
    assert result["full3_case_6_direction_denominators_retained"] is True
    assert result["source_sha256"] == RUNNER_SHA256 and result["protocol_sha256"] == PROTOCOL_SHA256
    assert result["expected_native_complete_sha256"] == NATIVE_COMPLETE_SHA256
    assert result["terminal_observation_sha256"] == TERMINAL_SHA256
    assert result["native_derived_numeric_inputs_used"] is True
    assert result["native_solution_outputs_used_during_own_solves"] is False
    assert result["canonical_input_manifest_sha256"] == INPUT_MANIFEST_SHA256
    assert result["canonical_input_source_sha256"] == INPUT_SOURCE_SHA256
    assert result["export_source_plan_sha256"] == EXPORT_PLAN_SHA256
    assert result["physical_acceptance"] is None
    assert [case["case_id"] for case in result["cases"]] == CASE_IDS

    cases, all_direction_errors = [], []
    for case in result["cases"]:
        assert case["status"] == "complete_own_solve" and case["comparison_status"] == "complete"
        assert _sha(output / case["artifact"]) == case["artifact_sha256"]
        record, comparison = case["record"], case["comparison"]
        assert "native_numeric_witnesses_used" not in record
        assert record["kernel_opens_native_files"] is False
        assert record["native_derived_numeric_inputs_used"] is True
        assert record["native_solution_outputs_used_during_own_solve"] is False
        canonical_row = next(row for row in manifest["cases"] if row["case"]["id"] == case["case_id"])
        assert record["canonical_input_array_sha256"] == canonical_row["array_sha256"]
        assert record["canonical_input_artifact_sha256"] == canonical_row["artifact_sha256"]
        assert case["canonical_input_artifact_sha256"] == canonical_row["artifact_sha256"]
        assert record["direction_context_m_sha256"] == canonical_row["array_sha256"]["direction_parameter_probe_m"]
        input_differences = _saved_input_differences(case, output)
        assert len(input_differences) == 4
        assert all(row["exact_value_identity"] for row in input_differences.values()), input_differences
        assert record["full_wavefield_history_saved"] is False
        assert record["time_bounds"] == {"time_m": 1, "time_M": 1999}
        assert record["receiver_samples_current_state"] is True
        assert comparison["trace_agreement"] is True and comparison["trace"]["finite"] is True
        trace_error, source_error = comparison["trace"]["value"], comparison["source_relative_L2"]["value"]
        assert trace_error is not None and math.isfinite(trace_error) and 0 <= trace_error <= 1e-9
        assert source_error is not None and math.isfinite(source_error) and 0 <= source_error <= 1e-14
        assert comparison["native_source_before_after"] is True
        assert comparison["native_physical_acceptance"] is None
        directions = case["directions"]
        assert [row["direction"] for row in directions] == ["interior", "edge"]
        errors = {}
        for direction in directions:
            assert direction["status"] == "complete"
            assert direction["informative"] is True and direction["agreement"] is True
            direction_input_error = direction["direction_relative_L2"]["value"]
            assert direction_input_error is not None and math.isfinite(direction_input_error)
            assert 0 <= direction_input_error <= 1e-14
            error = direction["relative_error"]
            assert error is not None and math.isfinite(error) and 0 <= error <= 1e-3
            errors[direction["direction"]] = error
            all_direction_errors.append(error)
        cases.append(dict(case_id=case["case_id"], trace_relative_L2=trace_error,
            source_relative_L2=source_error, informative_direction_relative_errors=errors))
    assert len(cases) == 3 and len(all_direction_errors) == 6
    report = dict(check="bp_numpy_tangent_reproduction", python=sys.version.split()[0],
        numpy=np.__version__, platform=platform.platform(), cases=cases,
        traces=3, informative_directions=6, trace_tolerance=1e-9, directional_tolerance=1e-3,
        maximum_trace_relative_L2=max(case["trace_relative_L2"] for case in cases),
        maximum_directional_relative_error=max(all_direction_errors),
        all_own_solves_before_native_solution_outputs=True,
        native_derived_numeric_inputs_used=True, native_solution_outputs_used_during_own_solves=False,
        independent_numeric_input_generation=False,
        canonical_input_manifest_sha256=INPUT_MANIFEST_SHA256, installed_devito_required=False,
        physical_acceptance=None,
        scope="known source-informed discrete fixture regression; no new physics, independent sites or student conclusions")
    with capsys.disabled():
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
