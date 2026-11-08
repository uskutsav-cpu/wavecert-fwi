"""Tiny synthetic analytic/math controls only; no actual fixture traces."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("green_candidate_test_unique", Path(__file__).with_name("compare_green.py"))
green = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green)


def test_gauss_polynomial_moments():
    x, w = green.gauss_legendre(4)
    for power in range(8):
        assert np.dot(w, x ** power) == pytest.approx(2 / (power + 1) if power % 2 == 0 else 0, abs=2e-14)


def test_synthetic_step_closed_form_and_causality():
    times = np.array([-1., 0., 1., 2., 2.5, 5.])
    nodes, weights = green.gauss_legendre(8)
    actual = green.convolve_green(times, 3., 1.5, 7., np.ones_like, nodes, weights)
    expected = green.step_closed_form(times, 3., 1.5, 7.)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    assert np.array_equal(actual[:4], np.zeros(4))


def test_strength_is_linear_and_units_scale_travel():
    times = np.array([3., 4.])
    x, w = green.gauss_legendre(6)
    a = green.convolve_green(times, 3., 1.5, 7., np.ones_like, x, w)
    b = green.convolve_green(times, 6., 3., 14., np.ones_like, x, w)
    np.testing.assert_allclose(b, 2 * a, rtol=1e-14)


def test_synthetic_ricker_peak_and_causal_cutoff():
    values = green.ricker_causal(np.array([-1., 0., 1., 2.]), 1., 1.)
    assert values[0] == 0 and values[2] == 1
    assert values[1] == values[3] < 0


def test_mask_excludes_uncomputed_sentinel_indices():
    times = np.arange(5.)
    mask = green.comparison_mask(times, 1, 3)
    assert mask.tolist() == [False, True, True, True, False]
    reference = np.array([99., 1., 2., 3., 99.])
    discrete = np.array([0., 1., 2., 3., 0.])
    result = green.trace_metrics(discrete, reference, times, mask, 2.)
    assert result["full_relative"]["value"] == 0
    assert result["sample_counts"] == dict(full=3, early=2, late=1)
    assert result["physical_quality_pass"] is None


def test_zero_denominator_has_no_pass_by_fiat():
    result = green.norm_ratio(np.ones(2), np.zeros(2))
    assert result["value"] is None and result["zero_denominator"]


@pytest.mark.parametrize("order", [0, -1, True, 2.5])
def test_invalid_orders(order):
    with pytest.raises(ValueError):
        green.gauss_legendre(order)


def test_default_cli_never_evaluates_or_loads_traces(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("No evaluation permitted")
    monkeypatch.setattr(green, "run", forbidden)
    monkeypatch.setattr(green, "convolve_green", forbidden)
    monkeypatch.setattr(green.np, "load", forbidden)
    assert green.main([]) == 0
    assert "metadata_only_no_trace_evaluation" in capsys.readouterr().out


def mock_run_inputs(monkeypatch):
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol, raw = green.load_protocol(protocol_path)
    expected = {**protocol["old_run_metadata_pins"], **protocol["old_raw_pins"]}
    monkeypatch.setattr(green, "old_snapshot", lambda *args: expected.copy())
    return protocol_path, green.sha(Path(green.__file__).read_bytes()), green.sha(raw)


def test_no_original_access_before_all_orders_saved(monkeypatch, tmp_path):
    protocol_path, source_sha, protocol_sha = mock_run_inputs(monkeypatch)
    def unfinished(*args):
        return np.arange(3.), np.array([False, True, False]), np.ones(3), np.ones(3)
    monkeypatch.setattr(green, "analytic_orders", unfinished)
    monkeypatch.setattr(green.np, "load", lambda *args, **kwargs: pytest.fail("must not open original traces"))
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="all three"):
        green.run(protocol_path, tmp_path / "original", output, source_sha, protocol_sha)
    status = json.loads((output / "status.json").read_bytes())
    assert status["status"] == "failed" and not status["comparison_started"]
    assert len(status["orders"]) == 3 and len(status["cases"]) == 7
    assert all(r["status"] == "unattempted" for r in status["orders"] + status["cases"])
    assert status["original_numeric_array_access"] == "not_started"
    assert not (output / "complete_receipt.json").exists()


def test_interrupted_order_attempt_retained(monkeypatch, tmp_path):
    protocol_path, source_sha, protocol_sha = mock_run_inputs(monkeypatch)
    def interrupted(protocol, output, status, persist):
        assert len(status["orders"]) == 3 and len(status["cases"]) == 7
        assert all(r["status"] == "unattempted" for r in status["orders"] + status["cases"])
        status["orders"][0]["status"] = "attempted"
        persist()
        assert json.loads((output / "status.json").read_bytes())["orders"][0]["status"] == "attempted"
        raise KeyboardInterrupt("synthetic interruption")
    monkeypatch.setattr(green, "analytic_orders", interrupted)
    output = tmp_path / "output"
    with pytest.raises(KeyboardInterrupt):
        green.run(protocol_path, tmp_path / "original", output, source_sha, protocol_sha)
    status = json.loads((output / "status.json").read_bytes())
    assert status["status"] == "failed" and status["orders"][0]["status"] == "interrupted"
    assert [r["order"] for r in status["orders"]] == [128, 256, 512]
    assert all(r["status"] == "unattempted" for r in status["orders"][1:] + status["cases"])
    assert not (output / "complete_receipt.json").exists()


def test_interrupted_first_case_keeps_all_seven_rows(monkeypatch, tmp_path):
    protocol_path, source_sha, protocol_sha = mock_run_inputs(monkeypatch)
    output = tmp_path / "output"
    def synthetic_orders(protocol, output, status, persist):
        for row in status["orders"]:
            row["status"] = "complete"
        status["all_three_analytic_orders_saved"] = True
        persist()
        return np.arange(3.), np.array([False, True, False]), np.ones(3), np.ones(3)
    def interrupted_load(*args, **kwargs):
        saved = json.loads((output / "status.json").read_bytes())
        assert len(saved["orders"]) == 3 and len(saved["cases"]) == 7
        assert saved["cases"][0]["status"] == "attempted"
        assert all(r["status"] == "unattempted" for r in saved["cases"][1:])
        raise KeyboardInterrupt("synthetic case interruption before decoding")
    monkeypatch.setattr(green, "analytic_orders", synthetic_orders)
    monkeypatch.setattr(green.np, "load", interrupted_load)
    with pytest.raises(KeyboardInterrupt):
        green.run(protocol_path, tmp_path / "original", output, source_sha, protocol_sha)
    saved = json.loads((output / "status.json").read_bytes())
    assert saved["status"] == "failed" and saved["cases"][0]["status"] == "interrupted"
    assert all(r["status"] == "unattempted" for r in saved["cases"][1:])
    assert not (output / "complete_receipt.json").exists()
