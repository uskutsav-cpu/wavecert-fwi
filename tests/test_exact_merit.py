import gc
import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import wavecert
from wavecert.physics import exact_merit
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.physics.residual_coded import best_exact_objective_gradient

PACKAGE = Path(wavecert.__file__).resolve().parent
ExactMeritOracle = exact_merit.ExactMeritOracle


def _problem(sources=(9, 12), receivers=(4, 5, 5, 7, 18)):
    p = Helmholtz2D(shape=(7, 8), spacing=.05, damping_width=2, damping_strength=3.7)
    rng = np.random.default_rng(448)
    m = rng.uniform(.19, .24, p.n)
    blocks = []
    for f in (2.3, 3.4):
        for source in sources:
            observed = p.restrict(p.solve_state(m*1.03, p.source(source), f), receivers)
            noise = rng.normal(size=len(receivers))+1j*rng.normal(size=len(receivers))
            observed = observed+.02*np.linalg.norm(observed)*noise/np.linalg.norm(noise)
            blocks.append(SurveyBlock(source, f, observed, f"{source}@{f}"))
    return p, m, blocks, list(receivers)


def _oracle(p, blocks, receivers):
    return ExactMeritOracle(p, blocks, receivers, reference_package_root=PACKAGE)


def _ordinary(p, m, blocks, receivers):
    objective, gradient = 0., np.zeros(p.n)
    for block in blocks:
        phi, g, _, _ = p.objective_and_gradient(m, p.source(block.source_index), block.observed,
            receivers, block.frequency_hz)
        objective += phi
        gradient += g
    return objective, gradient


@pytest.mark.parametrize("sources,receivers,route", [
    ((9, 12), (4, 5, 5, 7, 18), "forward_cache/source_adjoints"),
    ((9, 12), (9, 12, 9), "forward_cache/receiver_reuse"),
    ((9, 12, 18), (9, 12, 22, 25), "forward_cache/receiver_reuse"),
    ((9, 12, 18, 22, 25, 29), (7, 13, 7), "receiver_cache/residual_coded"),
    ((9, 12, 18, 22), (9, 12, 18, 12), "receiver_cache/source_states"),
    ((9, 12, 18, 22), (9, 12, 9), "receiver_cache/source_states"),
])
def test_all_incremental_routes_match_independent_adjoint_and_strongest_cold_counts(sources, receivers, route):
    p, model, blocks, receivers = _problem(sources, receivers)
    objective, gradient = _ordinary(replace(p), model, blocks, receivers)
    best = best_exact_objective_gradient(physics=replace(p), m=model, blocks=blocks, receiver_indices=receivers)
    oracle = _oracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    result = oracle.gradient(receipt)
    assert result.objective == pytest.approx(objective, rel=1e-12, abs=1e-12)
    assert np.allclose(result.gradient, gradient, rtol=1e-10, atol=1e-11)
    assert set(result.frequency_strategies.values()) == {route}
    expected_initial = 2*min(len(sources), len(set(receivers)))
    costs = oracle.cost_snapshot()
    assert costs["physics"]["forward_rhs"]+costs["physics"]["adjoint_rhs"] == best.expected_rhs
    assert expected_initial+result.additional_rhs == best.expected_rhs
    assert costs["physics"]["factorizations"] == 2
    assert costs["physics"]["svd_evaluations"] == costs["physics"]["dense_inverse_rhs"] == 0
    before = oracle.cost_snapshot()["physics"]
    again = oracle.gradient(receipt)
    assert again.additional_rhs == 0 and np.array_equal(again.gradient, result.gradient)
    assert before == oracle.cost_snapshot()["physics"]
    assert len(receipt.model_sha256) == len(receipt.operator_sha256) == len(receipt.geometry_sha256) == len(receipt.data_sha256) == 64
    json.dumps(receipt.to_dict(), allow_nan=False)
    json.dumps(oracle.contract, allow_nan=False)


def test_complex_damped_repeated_sampling_gradient_matches_objective_finite_difference():
    p, model, blocks, receivers = _problem((9, 12, 18, 22, 25, 29), (9, 12, 9))
    oracle = _oracle(p, blocks, receivers)
    result = oracle.gradient(oracle.evaluate(model))
    rng = np.random.default_rng(35)
    direction = rng.normal(size=p.n)
    direction /= np.linalg.norm(direction)

    def objective(values):
        return sum(p.objective(values, p.source(b.source_index), b.observed, receivers, b.frequency_hz) for b in blocks)

    h = 1e-5
    finite_difference = (objective(model+h*direction)-objective(model-h*direction))/(2*h)
    assert finite_difference == pytest.approx(result.gradient @ direction, rel=1e-6, abs=1e-9)


def test_accepted_fields_and_lu_survive_multiple_rejected_trials_without_a_repeated_forward_or_factorization():
    p, model, blocks, receivers = _problem()
    oracle = _oracle(p, blocks, receivers)
    current = oracle.evaluate(model)
    oracle.accept(current)
    trial1 = oracle.evaluate(model*1.001)
    oracle.evaluate(model*1.002)
    before = oracle.cost_snapshot()
    result = oracle.gradient(current)
    after = oracle.cost_snapshot()
    assert result.additional_rhs == len(blocks)
    assert after["physics"]["factorizations"] == before["physics"]["factorizations"] == 6
    assert after["physics"]["forward_rhs"] == before["physics"]["forward_rhs"] == 3*len(blocks)
    assert after["physics"]["adjoint_rhs"]-before["physics"]["adjoint_rhs"] == len(blocks)
    assert oracle.evaluate(model.copy()) is current
    oracle.accept(trial1)
    assert oracle.evaluate(model*1.001) is trial1
    # Main reference cache is intentionally disabled; reuse came from paid receipts.
    assert oracle._p.factorization_cache_size == 0 and not oracle._p._factorizations


def test_receipt_arrays_are_immutable_and_changed_model_bytes_require_new_paid_work():
    p, model, blocks, receivers = _problem()
    oracle = _oracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    g = oracle.gradient(receipt).gradient
    with pytest.raises(ValueError):
        receipt.model.setflags(write=True)
    with pytest.raises(ValueError):
        g.setflags(write=True)
    model[0] *= 1.01
    new = oracle.evaluate(model)
    assert new.model_sha256 != receipt.model_sha256 and new.key != receipt.key
    assert oracle.cost_snapshot()["physics"]["factorizations"] == 4


def test_independent_policy_oracles_cannot_share_paid_receipts():
    p, model, blocks, receivers = _problem()
    first, second = _oracle(p, blocks, receivers), _oracle(p, blocks, receivers)
    receipt = first.evaluate(model)
    with pytest.raises(ValueError, match="foreign"):
        second.gradient(receipt)
    with pytest.raises(ValueError, match="foreign"):
        second.accept(receipt)
    assert second.cost_snapshot()["physics"]["factorizations"] == 0


@pytest.mark.parametrize("sources,receivers,kind", [
    ((9, 12), (4, 5, 5, 7, 18), "forward_states"),
    ((9, 12, 18, 22), (9, 12, 9), "receiver_map"),
])
def test_issued_exports_preserve_complex_fields_affine_residuals_and_repeated_slot_sums_without_extra_rhs(sources, receivers, kind):
    p, model, blocks, receivers = _problem(sources, receivers)
    oracle = _oracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    oracle.accept(receipt)
    before = oracle.cost_snapshot()
    export = oracle.export_merit_fields(receipt)
    export.validate(oracle)
    assert export.receipt_key == receipt.key and export.model_sha256 == receipt.model_sha256
    assert export.physics_config == dict(shape=list(p.shape), spacing=p.spacing,
        damping_width=p.damping_width, damping_strength=p.damping_strength)
    assert np.array_equal(export.damping, p.damping)
    assert "ordinary floating point" in export.accuracy_scope
    assert "not analytically zero" in export.accuracy_scope
    for frame in export.frequencies:
        assert frame.cache_kind == kind and frame.transpose_symmetric
        assert frame.source_indices == sources and frame.receiver_indices == tuple(receivers)
        assert tuple(blocks[i].frequency_hz for i in frame.block_indices) == (frame.frequency_hz,)*len(sources)
        if kind == "forward_states":
            prediction = frame.cached_field[list(receivers)]
        else:
            prediction = (frame.source_amplitude*frame.cached_field[:, list(sources)])[frame.receiver_expand_indices]
            # Complex symmetric is not Hermitian: using C.H instead of C.T is wrong.
            direct = np.stack([p.solve_state(model, p.source(i), frame.frequency_hz) for i in sources], axis=1)
            assert np.allclose(prediction, direct[list(receivers)], rtol=1e-11, atol=1e-12)
            assert not np.allclose(prediction, (frame.source_amplitude*frame.cached_field.conj()[:, list(sources)])[frame.receiver_expand_indices], rtol=1e-4, atol=1e-8)
        assert np.allclose(frame.residual_slots, prediction-frame.observed_slots, rtol=1e-13, atol=1e-14)
        independently_summed = np.zeros_like(frame.residual_unique)
        for slot, unique_position in enumerate(frame.receiver_expand_indices):
            independently_summed[unique_position] += frame.residual_slots[slot]
        assert np.array_equal(frame.residual_unique, independently_summed)
        for name in ("cached_field", "observed_slots", "residual_slots", "residual_unique", "receiver_expand_indices"):
            with pytest.raises(ValueError):
                getattr(frame, name).setflags(write=True)
    for array in (export.model, export.damping):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    copy = export.physics_config
    copy["spacing"] = 99.
    assert export.physics_config["spacing"] == p.spacing
    json.dumps(export.to_dict(), allow_nan=False)
    after = oracle.cost_snapshot()
    assert after["physics"] == before["physics"]
    assert after["oracle"]["transpose_symmetry_checks"] == 2
    assert after["oracle"]["export_fingerprint_bytes"] > 0
    oracle.gradient(receipt)
    assert oracle.cost_snapshot()["oracle"]["transpose_symmetry_checks"] == 2


@pytest.mark.parametrize("change", ["copied_snapshot", "array_bytes", "metadata", "foreign_owner", "copied_receipt"])
def test_merit_exports_require_actual_issued_identity_and_unchanged_payload(change):
    p, model, blocks, receivers = _problem()
    oracle = _oracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    snapshot = oracle.export_merit_fields(receipt)
    before = oracle.cost_snapshot()["physics"]
    if change == "copied_snapshot":
        snapshot = replace(snapshot)
    elif change == "array_bytes":
        object.__setattr__(snapshot.frequencies[0], "cached_field", snapshot.frequencies[0].cached_field*1.01)
    elif change == "metadata":
        object.__setattr__(snapshot, "spacing", snapshot.spacing*1.01)
    elif change == "foreign_owner":
        oracle = _oracle(p, blocks, receivers)
        before = oracle.cost_snapshot()["physics"]
    else:
        with pytest.raises(ValueError, match="foreign"):
            oracle.export_merit_fields(replace(receipt))
        return
    with pytest.raises(ValueError, match="foreign|unissued|fingerprint"):
        oracle.validate_merit_fields(snapshot)
    assert oracle.cost_snapshot()["physics"] == before


def test_snapshot_does_not_keep_expired_receipt_lu_alive_and_mutated_data_invalidates_current_export():
    p, model, blocks, receivers = _problem()
    oracle = _oracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    snapshot = oracle.export_merit_fields(receipt)
    oracle.evaluate(model*1.001)  # Replaces unaccepted latest-trial receipt.
    del receipt
    gc.collect()
    with pytest.raises(ValueError, match="expired"):
        oracle.validate_merit_fields(snapshot)
    receipt = oracle.evaluate(model)
    snapshot = oracle.export_merit_fields(receipt)
    blocks[0].observed[0] += 1j
    before = oracle.cost_snapshot()["physics"]
    with pytest.raises(ValueError, match="changed"):
        oracle.validate_merit_fields(snapshot)
    assert before == oracle.cost_snapshot()["physics"]


@pytest.mark.parametrize("change", ["data", "receiver_order", "block_order", "operator", "configuration", "operator_method", "fractional_width", "complex_stiffness", "complex_damping", "nonfinite_damping"])
def test_mutated_contract_fails_closed_before_cached_merit_or_gradient_reuse(change):
    p, model, blocks, receivers = _problem()
    oracle = _oracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    if change == "data":
        blocks[0].observed[0] += 1.
    elif change == "receiver_order":
        receivers.reverse()
    elif change == "block_order":
        blocks.reverse()
    elif change == "operator":
        p.stiffness.data[0] += .1
    elif change == "operator_method":
        p.operator = lambda *args: None
    elif change == "fractional_width":
        p.damping_width += .5
    elif change == "complex_stiffness":
        p.__dict__["stiffness"] = p.stiffness.astype(complex)
        p.stiffness.data[0] += 1j*.1
    elif change == "complex_damping":
        p.__dict__["damping"] = p.damping.astype(complex)+.1j
    elif change == "nonfinite_damping":
        p.damping[0] = np.nan
    else:
        p.spacing *= 1.01
    before = oracle.cost_snapshot()["physics"]
    with pytest.raises(ValueError, match="changed"):
        oracle.evaluate(model)
    with pytest.raises(ValueError, match="changed"):
        oracle.gradient(receipt)
    assert oracle.cost_snapshot()["physics"] == before


def test_foreign_cached_reference_module_is_rejected(tmp_path, monkeypatch):
    module = importlib.import_module("wavecert.physics.helmholtz")
    p, _, blocks, receivers = _problem()
    foreign = tmp_path / "helmholtz.py"
    foreign.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(foreign))
    with pytest.raises(RuntimeError, match="foreign cached"):
        _oracle(p, blocks, receivers)


@pytest.mark.parametrize("invalid", ["nan_model", "complex_model", "model_shape", "bad_data", "duplicate_block", "bad_receiver"])
def test_invalid_inputs_are_not_silently_coerced_or_cached(invalid):
    p, model, blocks, receivers = _problem()
    if invalid == "bad_data":
        blocks[0].observed[0] = np.nan
    elif invalid == "duplicate_block":
        blocks.append(blocks[0])
    elif invalid == "bad_receiver":
        receivers[0] = True
    if invalid in {"bad_data", "duplicate_block", "bad_receiver"}:
        with pytest.raises(ValueError):
            _oracle(p, blocks, receivers)
        return
    oracle = _oracle(p, blocks, receivers)
    if invalid == "nan_model":
        model[0] = np.nan
    elif invalid == "complex_model":
        model = model.astype(complex)
    else:
        model = model.reshape(p.shape)
    with pytest.raises(ValueError):
        oracle.evaluate(model)
