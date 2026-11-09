"""Reproduce eight saved real-grid direction witnesses with NumPy alone."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_eight_initial_direction_raw_witnesses(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    run = root / "results" / "paid_merit_initial_direction_verification" / "run_01"
    script = run / "replay_raw_witnesses.py"
    expected = "1e9af84e461861c98dfc69f34d85d88c539b40f6658d3f1924cd32cb10729ccc"
    assert hashlib.sha256(script.read_bytes()).hexdigest() == expected
    output = tmp_path / "raw-replay.json"
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    subprocess.run(
        [sys.executable, "-I", str(script), "--execute-after-terminal-confirmation",
         "--run", str(run), "--output", str(output), "--expected-source-sha256", expected],
        check=True, capture_output=True, text=True, env=env, timeout=120,
    )
    result = json.loads(output.read_text())
    with capsys.disabled():
        print(json.dumps({
            "check": "eight_initial_raw_direction_witnesses", "python": sys.version.split()[0],
            "numpy": np.__version__, "platform": platform.platform(),
            "result": result["result"], "checks": result["checks"],
            "directions": result["directions_assessed"],
            "false_certifications": result["false_certification_count"],
            "literal_bound_misses": result["literal_bound_miss_count"],
            "cases": [{key: case[key] for key in (
                "case_index", "independent_gradient_relative_error",
                "approximate_gradient_relative_error", "true_directional_derivative",
                "approximate_center", "bound", "certified", "false_certification",
                "bound_covers_without_extra_allowance",
                "bound_covers_with_historical_helper_allowance",
            )} for case in result["cases"]],
        }), flush=True)
    assert result["result"] == "clear_raw_witness_replay"
    assert result["issues"] == []
    assert result["all8_case_slots_retained"] is True
    assert result["directions_assessed"] == len(result["cases"]) == 8
    assert result["false_certification_count"] == result["literal_bound_miss_count"] == 0
