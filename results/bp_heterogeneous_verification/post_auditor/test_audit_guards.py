"""Pure scalar and fake missing-run controls; no BP data or wave execution."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PRODUCER = HERE.with_name("bp_heterogeneous_guard_candidate")
spec = importlib.util.spec_from_file_location("_independent_bp_scalar_audit_test", HERE / "audit_guards.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def criteria():
    return json.loads((HERE / "criteria.json").read_text())


def test_scalar_reductions_retain_cancellation_and_objective_definition():
    assert audit.scalar_dot(np.array([1e16, 1., -1e16]), np.ones(3)) == 1
    assert audit.scalar_norm(np.array([3., 4.])) == 5
    mask = np.array([False, True, False])
    assert audit.objective(np.array([[100.], [3.], [100.]]), np.zeros((3, 1)), mask) == 4.5
    assert audit.ratio(np.zeros(2), np.zeros(2))["ratio"] is None


def test_scalar_edge_buckets_obey_extension_duality_and_differ_from_crop():
    field = np.arange(56.).reshape(7, 8) - 20
    physical = np.arange(12.).reshape(3, 4) / 11
    folded = audit.independent_fold(field, physical.shape, 2)
    assert audit.scalar_dot(field, np.pad(physical, 2, mode="edge")) == pytest.approx(audit.scalar_dot(folded, physical))
    assert not np.array_equal(folded, field[2:-2, 2:-2])


def test_independent_bilinear_field_and_exact_halfstep_mask():
    coarse = np.array([[1., 3.], [5., 7.]])
    np.testing.assert_array_equal(audit.bilinear_samples(coarse), [[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]])
    a, b = np.arange(0, 3.5, .5), np.arange(0, 3.25, .25)
    ma, mb = np.ones(len(a), bool), np.ones(len(b), bool)
    ma[[0, -1]] = False
    mb[[0, -1]] = False
    result = audit.common_comparison(dict(times_ms=a, computed_mask=ma, receiver=np.tile(a[:, None], (1, 10))),
                                     dict(times_ms=b, computed_mask=mb, receiver=np.tile(b[:, None], (1, 10))))
    assert result["aggregate"]["ratio"] == 0 and result["exact_common_samples"] == 5
    assert result["first_time_ms"] == .5 and result["last_time_ms"] == 2.5
    assert result["physical_acceptance"] is None


def test_scalar_close_is_finite_and_records_mismatches():
    collector = audit.Audit(criteria())
    assert collector.near("valid", 1., 1.)
    assert not collector.near("bad", 2., 1.)
    assert not collector.near("nonfinite", math.nan, 1.)
    assert len(collector.checks) == 3 and collector.checks[2]["absolute_error"] is None


def test_default_cli_does_not_decode_inputs_or_runs(monkeypatch, capsys):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array loading is forbidden"))
    monkeypatch.setattr(audit, "execute", lambda *a: pytest.fail("execution is forbidden"))
    assert audit.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "metadata_only_no_BP_or_native_output_reads" and result["planned_calls"] == 57


def fake_missing_run(tmp_path, complete=False):
    protocol = json.loads((PRODUCER / "protocol.json").read_text())
    native = tmp_path / "synthetic_missing_run"
    native.mkdir()
    terminal = tmp_path / "manufactured_terminal_evidence.txt"
    terminal.write_text("Unit-test-only terminal placeholder. No actual project run.\n")
    status = dict(status="complete_guard_attempts", all_case_workers_complete=False,
        calls=[dict(case_id=c, call_id=i, status="unattempted") for c, i in audit.planned(protocol)],
        case_processes=[dict(case_id=c["id"], status="unattempted") for c in protocol["inputs"]],
        source_sha256=criteria()["producer_sha256"], protocol_sha256=criteria()["protocol_sha256"], physical_acceptance=None)
    (native / "status.json").write_text(json.dumps(status))
    if complete:
        (native / "complete_receipt.json").write_text(json.dumps(status))
    return SimpleNamespace(producer=PRODUCER / "run_guards.py", terminal_receipt=terminal,
        expected_terminal_sha256=audit.checksum(terminal), output=tmp_path / "audit_output", run=native,
        repo=HERE.parents[1] / "wavecert-fwi", inputs=tmp_path / "NO_INPUTS",
        primary_source_root=Path("/opt/miniconda3/lib/python3.13/site-packages/examples/seismic"))


def test_missing_receipts_and_workers_preserve_all_denominators(monkeypatch, tmp_path):
    args = fake_missing_run(tmp_path)
    monkeypatch.setattr(audit, "source_digest", lambda *a: criteria()["production_source_sha256"])
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("No BP arrays may be loaded by this fake-missing-run test"))
    assert audit.execute(args, criteria()) == 1
    result = json.loads((args.output / "receipt.json").read_text())
    assert len(result["call_inventory"]) == 57 and len(result["cases"]) == 3
    assert all(row["native_status"] == "unattempted" for row in result["call_inventory"])
    assert sum(len(case["planned_derivatives"]) for case in result["cases"]) == 18
    assert result["integrity_checks_pass"] is False and result["physical_acceptance"] is None
    assert sum(case["status"] == "metadata_only_incomplete_native_receipt" for case in result["cases"]) == 3


def test_interrupt_retains57calls3cases18direction_rows(monkeypatch, tmp_path):
    args = fake_missing_run(tmp_path, complete=True)
    monkeypatch.setattr(audit, "source_digest", lambda *a: criteria()["production_source_sha256"])
    def interrupt(*args):
        raise KeyboardInterrupt("manufactured auditor interruption")
    monkeypatch.setattr(audit, "audit_case", interrupt)
    with pytest.raises(KeyboardInterrupt):
        audit.execute(args, criteria())
    result = json.loads((args.output / "status.json").read_text())
    assert result["status"] == "failed_or_interrupted" and len(result["call_inventory"]) == 57
    assert result["cases"][0]["status"] == "attempted" and result["cases"][1]["status"] == "unattempted"
    assert sum(len(case["planned_derivatives"]) for case in result["cases"]) == 18
    assert not (args.output / "receipt.json").exists()
