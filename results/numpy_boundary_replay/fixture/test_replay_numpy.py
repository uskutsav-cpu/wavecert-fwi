"""Exact moments, dense one-step algebra and guards only; no wave replay."""

import importlib.metadata
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import replay_numpy as module

PROTOCOL = Path(__file__).with_name("protocol.json")


def test_exact_moment_system_derives_classical_fourth_order_coefficients():
    weights = module.rational_weights()
    assert weights == (Fraction(-1, 12), Fraction(4, 3), Fraction(-5, 2), Fraction(4, 3), Fraction(-1, 12))
    for power in range(6):
        assert sum(w*Fraction(j)**power for j, w in zip(range(-2, 3), weights, strict=True)) == (2 if power == 2 else 0)


@pytest.mark.parametrize("variant", ["faithful_literals", "rational_spatial"])
def test_vectorized_one_step_matches_independent_dense_bands_and_one_sided_damping(variant):
    protocol, _, _, _ = module.load_protocol(PROTOCOL)
    weights = module.coefficients(variant, 10., protocol)
    far, near, central = weights
    shape, mass, timestep = (5, 6), .25, .7
    n = int(np.prod(shape))
    laplace = np.zeros((n, n))
    for x in range(shape[0]):
        for z in range(shape[1]):
            i = np.ravel_multi_index((x, z), shape)
            laplace[i, i] = -central
            for axis in (0, 1):
                for offset, coefficient in [(-2, -far), (-1, near), (1, near), (2, -far)]:
                    other = [x, z]
                    other[axis] += offset
                    if 0 <= other[axis] < shape[axis]:
                        laplace[i, np.ravel_multi_index(tuple(other), shape)] += coefficient
    rng = np.random.default_rng(229)
    current, previous = rng.normal(size=n), rng.normal(size=n)
    damp = np.linspace(0., .08, n)
    state = np.zeros((3, shape[0]+4, shape[1]+4))
    # t=5 means current slot2, previous slot1 and next slot0.
    state[2, 2:-2, 2:-2] = current.reshape(shape)
    state[1, 2:-2, 2:-2] = previous.reshape(shape)
    source, amplitude = (2, 3), .123
    denominator = np.diag(mass/timestep**2+damp/timestep)
    rhs = (2*mass/timestep**2+damp/timestep)*current-mass/timestep**2*previous+laplace@current
    expected = np.linalg.solve(denominator, rhs)
    expected[np.ravel_multi_index(source, shape)] += timestep**2/mass*amplitude
    current_slot, next_slot = module.one_step(state, 5, damp.reshape(shape), mass, timestep, weights, amplitude, source)
    assert (current_slot, next_slot) == (2, 0)
    np.testing.assert_allclose(state[next_slot, 2:-2, 2:-2].ravel(), expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(state[current_slot, 2:-2, 2:-2].ravel(), current)
    assert np.count_nonzero(state[:, :2]) == np.count_nonzero(state[:, -2:]) == 0
    assert np.count_nonzero(state[:, :, :2]) == np.count_nonzero(state[:, :, -2:]) == 0


def test_independent_damping_uses_declared_face_offsets_and_adds_corners():
    grid = module.damping((11, 13), 2, (10., 10.))
    coefficient = 1.5*np.log(1000.)/20.
    outer = coefficient*(1.5-np.sin(3*np.pi)/(2*np.pi))
    inner = coefficient*(1.-np.sin(2*np.pi)/(2*np.pi))
    assert grid[0, 0] == pytest.approx(2*outer)
    assert grid[1, 3] == pytest.approx(inner)
    assert grid[2, 3] == 0 and grid[-3, 3] == 0
    np.testing.assert_array_equal(grid[0], grid[-1])
    np.testing.assert_array_equal(module.damping((11, 13), 0, (10., 10.)), np.zeros((11, 13)))


def test_case_geometry_matches_fixed_padded_domains_and_exact_node_interpolation():
    _, _, original, _ = module.load_protocol(PROTOCOL)
    cases = {c["id"]: c for c in original["cases"]}
    for damped, undamped in [("small_nbl20", "matched_exterior_20_undamped"), ("small_nbl40", "matched_exterior_40_undamped")]:
        first = module.geometry(cases[damped], original["common"])
        second = module.geometry(cases[undamped], original["common"])
        assert first[:3] == second[:3]
        assert all(type(i) is int for i in first[1]+first[2])
    assert module.geometry(cases["reference_321"], original["common"])[0] == (401, 401)


def test_source_definition_is_independent_five_hz_ricker_with_declared_peak_and_tails():
    samples = module.ricker(np.array([0., 200., 400.]), .005)
    assert samples[1] == 1.
    assert samples[0] == samples[2] and samples[0] < 0
    assert samples[0] != 0  # no compact-support assumption.


def test_default_cli_does_not_propagate_or_open_reference_data(monkeypatch, capsys):
    monkeypatch.setattr(module, "solve_case", lambda *_: pytest.fail("propagation attempted"))
    monkeypatch.setattr(module, "compare_after_all_solves", lambda *_: pytest.fail("reference arrays opened"))
    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "metadata_only_no_propagation"


def test_interruption_retains_prefix_and_never_opens_reference_arrays(monkeypatch, tmp_path):
    output = tmp_path/"interrupted"
    monkeypatch.setattr(module, "source_receipt", lambda *_: {"scope": "mock hashes"})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda *_: SimpleNamespace(free=200*1024**2))
    monkeypatch.setattr(module, "compare_after_all_solves", lambda *_: pytest.fail("reference arrays opened before all own solves"))

    def interrupt(case, common, variant, protocol):
        status = json.loads((output/"status.json").read_text())
        assert len(status["records"]) == 1 and status["records"][0]["status"] == "attempted"
        assert not status["reference_numeric_arrays_opened"]
        raise KeyboardInterrupt("synthetic pre-propagation interrupt")

    monkeypatch.setattr(module, "solve_case", interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.run(PROTOCOL, tmp_path, tmp_path/"unopened_reference", output,
            module.sha(PROTOCOL.read_bytes()), module.sha(Path(module.__file__).read_bytes()))
    status = json.loads((output/"status.json").read_text())
    assert status["status"] == "failed" and status["records"][0]["status"] == "interrupted"
    assert not status["reference_numeric_arrays_opened"] and not (output/"complete_receipt.json").exists()


def test_wrong_runner_hash_stops_before_propagation_and_output_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "source_receipt", lambda *_: pytest.fail("source guard reached after wrong hash"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.run(PROTOCOL, tmp_path, tmp_path, tmp_path/"forbidden", "wrong", "wrong")
    assert not (tmp_path/"forbidden").exists()


def test_comparison_failure_marks_reference_access_unknown_after_all_mock_solves(monkeypatch, tmp_path):
    output = tmp_path/"comparison_failure"
    monkeypatch.setattr(module, "source_receipt", lambda *_: {"scope": "mock hashes"})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda *_: SimpleNamespace(free=200*1024**2))
    monkeypatch.setattr(module, "solve_case", lambda *_: ({"receiver": np.zeros((1, 1))}, {"scope": "mock; no propagation"}))

    def comparison(*_):
        status = json.loads((output/"status.json").read_text())
        assert len(status["records"]) == 14 and all(r["status"] == "complete" for r in status["records"])
        assert status["comparison_started"] and status["reference_numeric_arrays_opened"] is None
        raise OSError("synthetic comparison failure; no original arrays opened in test")

    monkeypatch.setattr(module, "compare_after_all_solves", comparison)
    with pytest.raises(OSError, match="synthetic comparison failure"):
        module.run(PROTOCOL, tmp_path, tmp_path/"unopened_reference", output,
            module.sha(PROTOCOL.read_bytes()), module.sha(Path(module.__file__).read_bytes()))
    status = json.loads((output/"status.json").read_text())
    assert status["status"] == "failed" and status["comparison_started"]
    assert status["reference_numeric_arrays_opened"] is None
    assert not (output/"complete_receipt.json").exists()


def test_source_guard_requires_copied_primary_assets_not_installed_devito(monkeypatch, tmp_path):
    protocol, _, _, _ = module.load_protocol(PROTOCOL)
    monkeypatch.setattr(module, "_production_digest", lambda *_: protocol["production_source_sha256"])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda *_: pytest.fail("installed package dependency accessed"))
    receipt = module.source_receipt(tmp_path, protocol, PROTOCOL.parent)
    assert len(receipt["copied_primary_sha256"]) == 5


def test_declared_comparison_tolerates_source_ulp_but_rejects_excitation_or_trace_errors(tmp_path):
    # Fresh synthetic helper inputs only: no original run, cached data or propagation.
    protocol, _, _, _ = module.load_protocol(PROTOCOL)
    reference_dir, output = tmp_path/"synthetic_reference", tmp_path/"synthetic_own"
    reference_dir.mkdir()
    output.mkdir()
    rows, references = [], []
    for i in range(3):
        name, filename = f"synthetic_{i}", f"case_{i}.npz"
        times, reference = np.arange(3.), np.ones((3, 1))
        source, receiver = reference.copy(), reference.copy()
        if i == 0:
            source[0] = np.nextafter(1., 2.)
        elif i == 1:
            source += 1e-6
        else:
            receiver += 2e-9
        np.savez(reference_dir/filename, receiver=reference, times_before_ms=times, source_before=reference)
        np.savez(output/filename, receiver=receiver, times_ms=times, source=source)
        references.append(dict(case_id=name, raw_artifact=filename,
            raw_sha256=module.sha((reference_dir/filename).read_bytes())))
        rows.append(dict(case_id=name, variant="faithful_literals", raw_artifact=filename))
    receipt = reference_dir/"complete_receipt.json"
    receipt.write_text(json.dumps(dict(case_records=references)))
    protocol["original_complete_receipt_sha256"] = module.sha(receipt.read_bytes())
    outcomes = module.compare_after_all_solves(output, reference_dir, rows, protocol)
    assert outcomes[0]["agreement"] and not outcomes[0]["source_samples_bitwise_equal"]
    assert 0 < outcomes[0]["source_relative_L2"] < 1e-14
    assert outcomes[0]["source_max_absolute_difference"] > 0
    assert not outcomes[1]["agreement"] and not outcomes[1]["matching_excitation"]
    assert not outcomes[2]["agreement"] and outcomes[2]["matching_excitation"]
    assert outcomes[2]["relative_L2"] > 1e-9
