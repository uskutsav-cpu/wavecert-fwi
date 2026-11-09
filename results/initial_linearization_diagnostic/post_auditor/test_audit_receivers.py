"""Pure manufactured scalar controls; no original inputs or outputs."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_receiver_scalar_audit_test", HERE / "audit_receivers.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_complex_real_contraction_and_independent_scalar_metrics():
    truth, initial, tangent = np.array([1 + 2j]), np.array([0j]), np.array([2 + 1j])
    value = audit.metrics([(truth, initial, tangent)])
    assert audit.real_product(np.array([1 + 2j]), np.array([3 + 4j])) == 11
    assert value["data_change_norm"] == pytest.approx(math.sqrt(5))
    assert value["tangent_norm"] == pytest.approx(math.sqrt(5))
    assert value["remainder_norm"] == pytest.approx(math.sqrt(2))
    assert value["real_alignment"] == pytest.approx(.8) and value["g_data_dot_delta"] == -4
    assert value["initial_objective"] == 2.5 and value["linearized_truth_step_objective"] == 1


def test_compensated_real_sum_preserves_cancellation():
    assert audit.real_product(np.array([1e16, 1., -1e16]), np.ones(3)) == 1


def test_zero_norms_remain_null_and_flagged():
    z = np.zeros((20, 6), dtype=np.complex128)
    value = audit.metrics([(z, z, z)])
    assert value["real_alignment"] is None and value["remainder_relative_to_data_change"] is None
    assert value["zero_data_change"] is True and value["zero_tangent"] is True


def test_bands_require_every_case_and_frequency():
    v = (np.ones((2, 1), dtype=np.complex128), np.zeros((2, 1), dtype=np.complex128), np.ones((2, 1), dtype=np.complex128))
    payload = {(case, frequency): v for case in range(2) for frequency in [.5, 1.]}
    assert audit.full_band(payload, range(2), [.5, 1.])["data_change_norm"] == pytest.approx(math.sqrt(8))
    del payload[1, 1.]
    assert audit.full_band(payload, range(2), [.5, 1.]) is None


def test_default_cli_reads_no_project_arrays_or_runs(monkeypatch, capsys):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array decoding forbidden"))
    monkeypatch.setattr(audit, "run", lambda *a: pytest.fail("audit execution forbidden"))
    assert audit.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "metadata_only_no_outcome_reads" and value["planned_frequency_rows"] == 64
