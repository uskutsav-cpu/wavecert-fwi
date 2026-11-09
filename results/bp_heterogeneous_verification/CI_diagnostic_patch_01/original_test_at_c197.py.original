"""Portable regression of three known BP-crop traces and six derivatives.

The isolated archived NumPy implementation completes all own solves before
opening native numeric witnesses. No installed Devito or native PDE run is
needed. These reused crops share one synthetic ancestor, not three sites.
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
FIXTURE = ARCHIVE / "numpy_replay"
INPUTS = ROOT / "results" / "bp2004_acquisition" / "run_01"
RUNNER_SHA256 = "c51e570934071eade9e59989cd57daddd064252cd930e410573db0365944d9e0"
PROTOCOL_SHA256 = "f0b9db69a434116f212182897f4fa34ffd714782cf0e8dd30adb0410cc139e8c"
NATIVE_COMPLETE_SHA256 = "ae2ca931876b6c1159af151394bfb0e9c0352ea0c384e879ac398e5bca019206"
TERMINAL_SHA256 = "75e54d29d1bf593471d9544363aa6c9d2b66ffa3db82d0aabe6e3d67c3fcd94d"
CASE_IDS = ["upper_left", "middle_center", "lower_right"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bp_numpy_traces_and_physical_m_tangents_without_devito(tmp_path, capsys):
    script, protocol_path = FIXTURE / "run_replay.py", FIXTURE / "protocol.json"
    native_run = ARCHIVE / "native_run_01"
    terminal_path = ARCHIVE / "native_terminal_observation_01.json"
    assert _sha(script) == RUNNER_SHA256
    assert _sha(protocol_path) == PROTOCOL_SHA256
    assert _sha(native_run / "complete_receipt.json") == NATIVE_COMPLETE_SHA256
    assert _sha(terminal_path) == TERMINAL_SHA256
    terminal = json.loads(terminal_path.read_text())
    assert terminal["exit_code"] == 0 and terminal["complete_receipt_sha256"] == NATIVE_COMPLETE_SHA256
    assert terminal["call_status_counts"] == {"complete": 57}
    protocol = json.loads(protocol_path.read_text())
    assert [case["id"] for case in protocol["inputs"]] == CASE_IDS
    assert protocol["criteria"] == dict(trace_relative_L2=1e-9, gradient_relative_error=1e-3,
        source_relative_L2=1e-14, direction_relative_L2=1e-14, fold_relative_L2=1e-13,
        near_zero_relative_normscale=1e-8)

    output = tmp_path / "bp-numpy-tangent-reproduction"
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    subprocess.run(
        [sys.executable, "-I", "-W", "error", str(script),
         "--protocol", str(protocol_path), "--execute-after-native-terminal",
         "--inputs", str(INPUTS), "--native-run", str(native_run),
         "--terminal-observation", str(terminal_path), "--output", str(output),
         "--expected-source-sha256", RUNNER_SHA256,
         "--expected-protocol-sha256", PROTOCOL_SHA256,
         "--expected-native-complete-sha256", NATIVE_COMPLETE_SHA256],
        check=True, capture_output=True, text=True, env=env, timeout=300,
    )
    result = json.loads((output / "complete_receipt.json").read_text())
    assert json.loads((output / "status.json").read_text()) == result
    assert result["status"] == "complete_comparison"
    assert result["result"] == "ALL_FIXED_NUMERICAL_CHECKS_AGREE"
    assert result["numerical_agreement_pass"] is True
    assert result["own_solves_planned"] == 3 and result["directions_planned"] == 6
    assert result["all3_own_complete_before_native_arrays"] is True
    assert result["native_comparison_started"] is True and result["native_numeric_arrays_opened"] is True
    assert result["full3_case_6_direction_denominators_retained"] is True
    assert result["source_sha256"] == RUNNER_SHA256 and result["protocol_sha256"] == PROTOCOL_SHA256
    assert result["expected_native_complete_sha256"] == NATIVE_COMPLETE_SHA256
    assert result["terminal_observation_sha256"] == TERMINAL_SHA256
    assert result["physical_acceptance"] is None
    assert [case["case_id"] for case in result["cases"]] == CASE_IDS

    cases, all_direction_errors = [], []
    for case in result["cases"]:
        assert case["status"] == "complete_own_solve" and case["comparison_status"] == "complete"
        assert _sha(output / case["artifact"]) == case["artifact_sha256"]
        record, comparison = case["record"], case["comparison"]
        assert record["native_numeric_witnesses_used"] is False
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
        all_own_solves_before_native_numeric_witnesses=True, installed_devito_required=False,
        physical_acceptance=None,
        scope="known source-informed discrete fixture regression; no new physics, independent sites or student conclusions")
    with capsys.disabled():
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
