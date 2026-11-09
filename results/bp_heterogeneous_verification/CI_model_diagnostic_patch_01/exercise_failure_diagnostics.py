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



# Two unequal entries at exactly one and three representable positive-float steps.
own = module.np.array([[1., 2.], [3., 4.]])
native = own.copy()
native.view("<u8")[0, 0] += 1
native.view("<u8")[1, 0] += 3
comparison = module._model_array_difference(own, native)
assert comparison["unequal_count"] == 2 and comparison["maximum_ULP_distance"] == 3
assert comparison["own_sha256"] != comparison["native_sha256"]
assert "comparison_unavailable" in module._model_array_difference(own[:1], native)
assert "comparison_unavailable" in module._model_array_difference(own * 0, native)

output = HERE / "synthetic_bound_models"
output.mkdir()
native_run = HERE / "synthetic_native"
(native_run / "upper_left").mkdir(parents=True)
module.np.savez(output / "own.npz", physical_probe_velocity=own, physical_probe_m=own)
module.np.savez(native_run / "upper_left/probe.npz", actual_physical_velocity_kmps=own, actual_physical_m=native)
receipt = dict(calls=[dict(case_id="upper_left", call_id="probe_saved", result=dict(
    artifact="probe.npz", artifact_sha256=module._sha(native_run / "upper_left/probe.npz")))])
(native_run / "complete_receipt.json").write_text(json.dumps(receipt))
module.NATIVE_COMPLETE_SHA256 = module._sha(native_run / "complete_receipt.json")
row = dict(case_id="upper_left", artifact="own.npz", artifact_sha256=module._sha(output / "own.npz"))
differences = module._saved_model_differences(row, output, native_run)
assert differences["physical_velocity"]["maximum_ULP_distance"] == 0
assert differences["physical_m"]["maximum_ULP_distance"] == 3
# Deliberately stale artifact hash: the original exit2 diagnostics must survive.
with (output / "own.npz").open("ab") as stream:
    stream.write(b"synthetic stale bytes")
(output / "status.json").write_text(json.dumps(dict(
    all3_own_complete_before_native_arrays=True, cases=[row])))
detail = json.loads(module._replay_failure_diagnostics(completed, output, native_run))
assert detail["returncode"] == 2 and detail["stdout"] == completed.stdout
assert "saved_model_diagnostic_error" in detail["cases"][0]
# The numerical witness phase guard prevents diagnostic array reads too.
(output / "status.json").write_text(json.dumps(dict(
    all3_own_complete_before_native_arrays=False, cases=[row])))
detail = json.loads(module._replay_failure_diagnostics(completed, output, native_run))
assert "saved_models" not in detail["cases"][0]
assert "saved_model_diagnostic_error" not in detail["cases"][0]
print(json.dumps(dict(result="PASS", controls=["nonzero_still_fails_with3cases6directions", "missing_status", "malformed_status", "wrong_schema", "positive_float_ULP1and3", "shape_and_positive_guards", "pinned_NPZ_model_binding", "stale_artifact_caught", "phase_guard_prevents_model_read"], replay_or_native_execution=False)))
