"""Synthetic exit2/status-message control; never invokes a replay subprocess."""
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[1] / "wavecert-fwi/tests/test_bp_numpy_tangent_reproduction.py"
spec = importlib.util.spec_from_file_location("_bp_ci_diagnostic_test_only", TEST)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
completed = subprocess.CompletedProcess([], 2, stdout="synthetic stdout marker", stderr="synthetic stderr marker")
original_run = module.subprocess.run


def fake_run(arguments, **kwargs):
    assert kwargs["check"] is False
    output = Path(arguments[arguments.index("--output") + 1])
    output.mkdir(parents=True)
    cases = [dict(case_id=name, status="complete_own_solve", comparison_status="failed",
        comparison_error="synthetic comparison mismatch", comparison_prefix=dict(trace=dict(value=.25)),
        directions=[dict(direction=d, status="unavailable", relative_error=None) for d in ["interior", "edge"]])
        for name in module.CASE_IDS]
    status = dict(status="complete_comparison", native_numeric_arrays_opened=True,
        all3_own_complete_before_native_arrays=True, numerical_agreement_pass=False, cases=cases)
    (output / "status.json").write_text(json.dumps(status))
    return completed


try:
    module.subprocess.run = fake_run
    try:
        module.test_bp_numpy_traces_and_physical_m_tangents_without_devito(HERE / "synthetic_nonzero", None)
    except AssertionError as exc:
        detail = json.loads(str(exc))
        assert detail["returncode"] == 2 and len(detail["cases"]) == 3
        assert detail["stdout"] == completed.stdout and detail["stderr"] == completed.stderr
        assert all(c["comparison_error"] == "synthetic comparison mismatch" and c["trace"]["value"] == .25 for c in detail["cases"])
        assert sum(len(c["directions"]) for c in detail["cases"]) == 6
    else:
        raise AssertionError("nonzero child exit incorrectly passed")
finally:
    module.subprocess.run = original_run

for name, payload in [("missing", None), ("malformed", "{not json"), ("wrong_schema", "[]")]:
    output = HERE / ("synthetic_" + name)
    output.mkdir()
    if payload is not None:
        (output / "status.json").write_text(payload)
    detail = json.loads(module._replay_failure_diagnostics(completed, output))
    assert detail["stdout"] == completed.stdout and detail["stderr"] == completed.stderr
    assert "saved_status_error" in detail

print(json.dumps(dict(result="PASS", controls=["nonzero_still_fails_with3cases6directions", "missing_status", "malformed_status", "wrong_schema"], replay_or_native_execution=False)))
