"""Actual byte-frozen producer exports, scalar mocks only, in a subprocess."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import audit_retained
import pytest
import trace_checks

HERE = Path(__file__).resolve().parent
PRODUCER = HERE.parent / "fractional_velocity_prior48_driver_candidate"


@pytest.mark.parametrize("weight", [0.0, 100.0])
def test_actual_nonzero_query_frozen_engine_export_matches_independent_schema(tmp_path, weight):
    before = {
        name
        for name in sys.modules
        if name.startswith("wavecert") or name.startswith("_test_actual_frozen")
    }
    output = tmp_path / ("scalar_export_" + str(int(weight)))
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            str(HERE / "export_manufactured_producer_schema.py"),
            "--producer-directory",
            str(PRODUCER),
            "--output",
            str(output),
            "--weight",
            str(weight),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, dict(stdout=result.stdout, stderr=result.stderr)
    payload_path = output / "actual_exported_payload.json"
    metadata = json.loads((output / "export_metadata.json").read_text())
    assert hashlib.sha256(payload_path.read_bytes()).hexdigest() == metadata["payload_sha256"]
    payload = json.loads(payload_path.read_bytes())
    checks = audit_retained.Checks()
    criteria = json.loads((HERE / "criteria.json").read_bytes())
    audited = trace_checks.audit_engine(payload["engine"], checks, criteria)
    assert not checks.failures, checks.failures
    assert audited["method_valid"] is True
    assert audited["last_eligible"]["model_sha256"] == audit_retained.model_sha(
        payload["final_model"]
    )
    assert metadata["library_events"] > 0 and metadata["callback_point_records"] > 0
    assert metadata["actual_scalar_queries"]["gradient_dispatches"] > 0
    assert (
        metadata["actual_frozen_engine_called"] is True
        and metadata["actual_frozen_adapter_called"] is True
    )
    assert metadata["actual_driver_execute_policy_called"] is True
    assert metadata["actual_SciPy_minimize_used"] is True
    assert metadata["project_PDE_calls"] == metadata["project_array_reads"] == 0
    assert metadata["fake_clock_for_schema_only"] is True
    assert len(payload["initial_model"]) == len(payload["final_model"]) == 1
    assert (
        payload["initial_native_header"]["native_data_receipt_header"]["kind"]
        == "manufactured_scalar_schema_receipt_NOT_NATIVE_PDE"
    )
    after = {
        name
        for name in sys.modules
        if name.startswith("wavecert") or name.startswith("_test_actual_frozen")
    }
    assert after == before
