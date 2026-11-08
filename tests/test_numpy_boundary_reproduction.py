"""Portable numerical reproduction of seven archived finite-grid traces.

No installed Devito is needed. The reference-informed NumPy implementation
computes all seven own traces before this test opens a target numeric array.
"""

import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO/"results/numpy_boundary_replay/fixture"
TARGET = REPO/"results/devito_boundary_verification/run_01"
RUNNER_SHA256 = "4832612882cb7b484777e309fb14ccc291685ad1afb5efd030332d84df31375c"
PROTOCOL_SHA256 = "53fbdd57a119ec68dd07e8c3ce833857e281acce13c0ff9001456ec114b59550"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_module():
    path = (FIXTURE/"replay_numpy.py").resolve()
    assert _sha(path) == RUNNER_SHA256, "archived NumPy implementation bytes changed"
    name = "_wavecert_numpy_boundary_reproduction_48326128"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    assert Path(module.solve_case.__code__.co_filename).resolve() == path
    return module


def _relative_l2(own, target):
    if own.shape != target.shape or not np.all(np.isfinite(own)) or not np.all(np.isfinite(target)):
        return None
    denominator = float(np.linalg.norm(target))
    return None if denominator == 0 else float(np.linalg.norm(own-target))/denominator


def test_faithful_numpy_boundary_reproduction_without_devito(capfd, record_property):
    protocol_path = FIXTURE/"protocol.json"
    assert _sha(protocol_path) == PROTOCOL_SHA256, "archived replay protocol bytes changed"
    module = _load_frozen_module()
    protocol, _, original, _ = module.load_protocol(protocol_path)
    assert protocol["agreement_relative_L2_tolerance"] == 1e-9
    assert protocol["matching_excitation_relative_L2_tolerance"] == 1e-14
    assert protocol["time_bounds"]["time_m"] == 1 and protocol["time_bounds"]["time_M"] == 1099
    assert len(original["cases"]) == 7

    # This entire loop finishes before any target NPZ (or target receipt) opens.
    own_traces = []
    for case in original["cases"]:
        arrays, record = module.solve_case(case, original["common"], "faithful_literals", protocol)
        assert record["finite"] and not record["original_receiver_or_source_arrays_used"]
        own_traces.append((case["id"], arrays))
    assert len(own_traces) == 7

    receipt_path = TARGET/"complete_receipt.json"
    assert _sha(receipt_path) == protocol["original_complete_receipt_sha256"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "complete_software_verification"
    targets = {row["case_id"]: row for row in receipt["case_records"]}
    assert set(targets) == {name for name, _ in own_traces}
    diagnostics = []
    for name, own in own_traces:
        row = targets[name]
        assert row["status"] == "complete"
        path = TARGET/row["raw_artifact"]
        assert _sha(path) == row["raw_sha256"]
        with np.load(path, allow_pickle=False) as target:
            same_time = bool(np.array_equal(own["times_ms"], target["times_before_ms"]))
            source_error = _relative_l2(own["source"], target["source_before"])
            trace_error = _relative_l2(own["receiver"], target["receiver"])
            source_bitwise = bool(np.array_equal(own["source"], target["source_before"]))
        diagnostics.append(dict(case_id=name, exact_time_axis=same_time,
            source_relative_L2=source_error, trace_relative_L2=trace_error,
            source_bitwise_diagnostic=source_bitwise,
            agreement=bool(same_time and source_error is not None and source_error <= 1e-14
                           and trace_error is not None and trace_error <= 1e-9)))

    report = dict(check="seven_faithful_NumPy_traces", python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(),
        trace_tolerance=1e-9, excitation_tolerance=1e-14,
        maximum_trace_relative_L2=max((r["trace_relative_L2"] for r in diagnostics if r["trace_relative_L2"] is not None), default=None),
        maximum_source_relative_L2=max((r["source_relative_L2"] for r in diagnostics if r["source_relative_L2"] is not None), default=None),
        all_own_traces_computed_before_targets=True, installed_devito_required=False,
        scope="reference-informed independent implementation replay; no continuum, student-ownership or goal-completion claim",
        cases=diagnostics)
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    record_property("numpy_boundary_reproduction", encoded)
    # Preserve the per-case numerical evidence in the existing CI console log.
    with capfd.disabled():
        print(encoded)
    assert all(row["agreement"] for row in diagnostics), encoded
