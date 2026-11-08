"""Cross-platform numerical regressions of known diagnostic fixtures.

These reproduce fixed software evidence, not new confirmation or physics passes.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def execute(script, source_hash, protocol_hash, output, arguments):
    assert hashlib.sha256(script.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(script.with_name("protocol.json").read_bytes()).hexdigest() == protocol_hash
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    subprocess.run(
        [sys.executable, "-I", str(script), *arguments, "--output", str(output),
         "--expected-source-sha256", source_hash, "--expected-protocol-sha256", protocol_hash],
        check=True, capture_output=True, text=True, env=env, timeout=120,
    )


def test_initial_recovery_metric_slopes(tmp_path, capsys):
    fixture = ROOT / "results" / "initial_recovery_slope_diagnostic" / "fixture"
    output = tmp_path / "recovery-diagnostic"
    execute(
        fixture / "diagnose_initial_slopes.py",
        "7998631fc3542cdd42ebce0bcc236d87755a33647b1135327ebebda6f8dec46e",
        "b1104b4159ffc0065fb5b3a47a1c552a8cb94fc329daa8f7b315418452d7a8d2",
        output,
        ["--execute-reviewed-diagnostic", "--repo", str(ROOT), "--witness",
         str(ROOT / "results" / "paid_merit_initial_direction_verification" / "run_01"),
         "--benchmark", str(ROOT / "results" / "paid_merit_engineering_benchmark" / "run_01")],
    )
    result = json.loads((output / "summary.json").read_text())
    cases = [json.loads((output / f"case_{index:02d}_slopes.json").read_text()) for index in range(8)]
    with capsys.disabled():
        print(json.dumps(dict(
            check="recovery_metric_initial_slopes", python=sys.version.split()[0], numpy=np.__version__,
            platform=platform.platform(), result=result["result"], checks=result["checks"],
            truth_feasible=result["truth_models_literal_box_feasible_count"],
            initial_feasible=result["initial_models_literal_box_feasible_count"],
            directions=result["direction_summaries"],
            toward_truth_data_slope=result["toward_truth_data_slope_numerical_sign_counts"],
            endpoint_instances=result["original_internal_endpoint_instances"],
            cases=[dict(case_index=case["case_index"],
                        toward_truth_dataDD=case["data_gradient_dot_toward_truth_correction"]["value"],
                        directions={name: dict(
                            dataDD=row["data_directional_derivative"]["value"],
                            m_error_slope=row["half_relative_squared_metric_derivatives"]["half_relative_squared_m_error"]["analytic"],
                            velocity_error_slope=row["half_relative_squared_metric_derivatives"]["half_relative_squared_velocity_error"]["analytic"],
                        ) for name, row in case["directions"].items()}) for case in cases],
        )), flush=True)
    assert result["result"] == "clear_software_diagnostic"
    assert result["issues"] == [] and result["all8_cases_retained"] is True
    assert result["truth_models_literal_box_feasible_count"] == result["initial_models_literal_box_feasible_count"] == 8
    assert result["original_internal_endpoint_instances"] == 1440
    assert result["new_PDE_solves"] == result["new_optimizer_trajectories"] == 0
    for direction in result["direction_summaries"].values():
        assert direction["available_directions"] == direction["full_case_denominator"] == 8
        assert direction["metrics"]["half_relative_squared_velocity_error"]["literal_local_opposition_cases"] == 5
        assert direction["metrics"]["half_relative_squared_m_error"]["literal_local_opposition_cases"] == 3


def test_analytical_homogeneous_wave_reference(tmp_path, capsys):
    fixture = ROOT / "results" / "analytic_green_verification" / "fixture"
    output = tmp_path / "green-diagnostic"
    execute(
        fixture / "compare_green.py",
        "ba7b03027d837f6e834e47ebf13c79768287d6d2963b1df3b898a6cbce822b69",
        "729e23e1a8a5373ba5492a47f2259d0bcb9a0618b25ddc87f3d73e0ea44337e5",
        output,
        ["--execute-analytical-comparison", "--original-run",
         str(ROOT / "results" / "devito_boundary_verification" / "run_01")],
    )
    result = json.loads((output / "status.json").read_text())
    with capsys.disabled():
        print(json.dumps(dict(
            check="analytic_homogeneous_wave_reference", python=sys.version.split()[0], numpy=np.__version__,
            platform=platform.platform(), status=result["status"],
            internal_numeric_gates=result["internal_numeric_gates_pass"],
            physical_quality_pass=result["physical_quality_pass"],
            quadrature_convergence=result["quadrature_convergence"],
            cases=[dict(case_id=row["case_id"], status=row["status"],
                        full_relative_L2=row["metrics"]["full_relative"]["value"],
                        early_relative_L2=row["metrics"]["early_relative"]["value"],
                        sample_counts=row["metrics"]["sample_counts"])
                   for row in result["cases"]],
        )), flush=True)
    assert result["status"] == "complete_analytical_diagnostic"
    assert result["internal_numeric_gates_pass"] is True
    assert result["all_input_software_gates_pass"] is True
    assert result["source_and_original_unchanged"] is True
    assert result["physical_quality_pass"] is None
    assert [row["order"] for row in result["orders"]] == [128, 256, 512]
    assert len(result["cases"]) == 7 and all(row["status"] == "complete" for row in result["cases"])
