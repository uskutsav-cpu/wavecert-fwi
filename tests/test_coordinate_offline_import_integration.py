"""Portable manufactured regression for the imported auditor API boundary."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_coordinate_offline_import_integration(capsys):
    fixture = ROOT / "results" / "coordinate_cost_engineering_comparison" / "fixture_v2"
    pins = {
        "run_coordinate_comparison.py": "351b527170751d958939dfbc1f17b091770696ba043cf522be0db5f238e19c63",
        "independent_saved_models.py": "807f4f2f0e0aeea1d2f61b64ee5f64a99819c38d4f6164529e664f0a235edeae",
        "protocol.json": "f0306803ed34c03318e9a64d2848a8ce57900a6b4e5998c05d88d2bcbfa523f4",
    }
    for name, expected in pins.items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == expected
    code = r'''
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

fixture = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("coordinate_ci_driver", fixture / "run_coordinate_comparison.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
audit = driver.import_file("coordinate_ci_real_auditor", fixture / "independent_saved_models.py")
assert not hasattr(audit, "np")
driver.initialize_independent_runtime(audit)
assert all(getattr(audit, key) is not None for key in ("np", "scipy", "ndi", "sp", "spla"))
geometry = dict(original_source_indices=[50], original_receiver_indices=[54, 58], frequencies_hz=[2.5])
synthetic = dict(geometry=geometry)
checks = audit.Checks()
raw = np.full((24, 24), .2)
case = audit.IndependentCase(raw, synthetic, checks, case_index=0)
initial, final = case.initial.copy(), case.initial * 1.01
context = dict(raw_case=dict(raw_model=raw), prepared=dict(truth=case.truth.copy()),
    common_initial_m=initial, final_native_m=final,
    result=dict(initial_objective=case.evaluate(initial)["objective"],
                final_objective=case.evaluate(final)["objective"]))
protocol = json.loads((fixture / "protocol.json").read_bytes())
result = driver.offline_audit(dict(case_index=0), context, synthetic,
                            SimpleNamespace(audit=audit, np=np), protocol)
assert result["failures"] == [] and result["numeric_comparisons"] == 6
assert result["maximum_relative_linear_residual"] <= protocol["offline_audit"]["linear_residual_maximum"]
initial_error = result["own_native_model_errors"]["common_initial"]
final_error = result["own_native_model_errors"]["returned_final"]
assert initial_error["relative_squared_slowness_L2"] < 1e-11
assert initial_error["relative_physical_velocity_L2"] < 1e-11
assert abs(final_error["relative_squared_slowness_L2"] - .01) < 1e-11
assert abs(final_error["relative_physical_velocity_L2"] - (1 - 1 / math.sqrt(1.01))) < 1e-11
print(json.dumps(dict(check="coordinate_offline_import_integration", python=sys.version.split()[0],
    numpy=np.__version__, platform=platform.platform(), synthetic_shape=[24, 24],
    refined_shape=[49, 49], frequencies=1, sources=1, receivers=2,
    checks=result["checks"], numeric_comparisons=result["numeric_comparisons"],
    initial_error=initial_error, final_error=final_error,
    maximum_relative_linear_residual=result["maximum_relative_linear_residual"],
    scope="Manufactured API regression; no project arrays, benchmark trajectories or scientific confirmation")))
'''
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    result = subprocess.run([sys.executable, "-I", "-c", code, str(fixture)],
                            check=True, capture_output=True, text=True, env=env, timeout=60)
    report = json.loads(result.stdout)
    assert report["check"] == "coordinate_offline_import_integration"
    with capsys.disabled():
        print(json.dumps(report), flush=True)
