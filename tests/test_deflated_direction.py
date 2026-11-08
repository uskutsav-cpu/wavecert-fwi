import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la

from wavecert.certificates.deflated_direction import DeflatedDirectionCertifier
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock


def case(frequencies=(1.0, 1.3), damping=2.0):
    physics = Helmholtz2D(shape=(5, 7), spacing=0.12, damping_width=2, damping_strength=damping)
    model = np.linspace(0.2, 0.4, physics.n)
    receivers = (2, 3, 3, 4, 5, 6)
    rng = np.random.default_rng(831)
    blocks = [
        SurveyBlock(
            source,
            frequency,
            0.2 * rng.normal(size=6) + 0.1j * rng.normal(size=6),
            f"{source}@{frequency}",
        )
        for source in (8, 12)
        for frequency in frequencies
    ]
    return physics, model, receivers, blocks


def explicit_basis(physics, rank):
    nz, nx = physics.shape
    z, x = np.arange(1, nz + 1)[:, None], np.arange(1, nx + 1)[None, :]
    modes = []
    for k in range(1, nz + 1):
        for p in range(1, nx + 1):
            eigenvalue = (
                4
                / physics.spacing**2
                * (
                    np.sin(np.pi * k / (2 * (nz + 1))) ** 2
                    + np.sin(np.pi * p / (2 * (nx + 1))) ** 2
                )
            )
            vector = (
                2
                / np.sqrt((nz + 1) * (nx + 1))
                * np.sin(np.pi * k * z / (nz + 1))
                * np.sin(np.pi * p * x / (nx + 1))
            )
            modes.append((eigenvalue, k, p, vector.reshape(-1)))
    modes.sort(key=lambda row: row[:3])
    return np.column_stack([r[3] for r in modes[:rank]]), modes


def manual_fields(physics, model, receivers, block, rank, sweeps):
    basis, modes = explicit_basis(physics, rank)
    matrix = physics.operator(model, block.frequency_hz).toarray()
    ar = basis.T @ matrix @ basis
    omega = 2 * np.pi * block.frequency_hz
    midpoint_m = (np.min(model) + np.max(model)) / 2
    midpoint_d = (np.min(physics.damping) + np.max(physics.damping)) / 2
    high = sum(
        np.outer(row[3], row[3]) / (row[0] - omega**2 * midpoint_m + 1j * omega * midpoint_d)
        for row in modes[rank:]
    )
    high_h = high.conjugate()
    m_action = high + basis @ la.solve(ar, basis.T @ (np.eye(physics.n) - matrix @ high))
    n_action = high_h + basis @ la.solve(
        ar.conjugate().T, basis.T @ (np.eye(physics.n) - matrix.conjugate().T @ high_h)
    )
    q = physics.source(block.source_index)
    state = basis @ la.solve(ar, basis.T @ q)
    rhs = physics.inject_receivers(state[list(receivers)] - block.observed, receivers)
    adjoint = basis @ la.solve(ar.conjugate().T, basis.T @ rhs)
    for _ in range(sweeps):
        state += m_action @ (q - matrix @ state)
        rhs = physics.inject_receivers(state[list(receivers)] - block.observed, receivers)
        adjoint += n_action @ (rhs - matrix.conjugate().T @ adjoint)
    return state, adjoint


@pytest.mark.parametrize("sweeps", [0, 1, 2, 4])
def test_fixed_coupled_updates_match_independent_dense_sine_decomposition(sweeps):
    p, model, receivers, blocks = case()
    result = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=sweeps).certify_survey(
        model, blocks, receivers
    )
    for block, computed in zip(blocks, result.block_certificates, strict=True):
        state, adjoint = manual_fields(p, model, receivers, block, 18, sweeps)
        assert np.allclose(computed.state, state, rtol=3e-11, atol=1e-12)
        assert np.allclose(computed.adjoint, adjoint, rtol=3e-11, atol=1e-12)
    assert result.costs["certificate"].get("paired_repair_block_sweeps", 0) == sweeps * len(blocks)
    assert result.costs["certificate"].get("high_inverse_rhs_columns", 0) == 2 * sweeps * len(
        blocks
    )
    assert all(value == 0 for value in result.costs["reference_solver"].values())
    json.dumps(result.to_dict(), allow_nan=False)


def test_available_bounds_cover_independent_fine_gradients_and_fd():
    p, model, receivers, blocks = case()
    result = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=4).certify_survey(
        model, blocks, receivers
    )
    exact = [
        p.objective_and_gradient(
            model, p.source(b.source_index), b.observed, receivers, b.frequency_hz
        )
        for b in blocks
    ]
    gradient = sum(row[1] for row in exact)
    derivative = float(gradient @ result.direction)
    assert result.available and result.certified_descent and derivative < 0
    assert (
        abs(derivative - result.approximate_directional_derivative)
        <= result.directional_error_bound + 1e-10
    )

    def objective(m):
        return sum(
            p.objective(m, p.source(b.source_index), b.observed, receivers, b.frequency_hz)
            for b in blocks
        )

    for step in (1e-3, 3e-4):
        numerical = (
            -objective(model + 2 * step * result.direction)
            + 8 * objective(model + step * result.direction)
            - 8 * objective(model - step * result.direction)
            + objective(model - 2 * step * result.direction)
        ) / (12 * step)
        assert derivative == pytest.approx(numerical, rel=1e-6, abs=1e-9)


def test_no_online_fine_solves_and_actual_fft_column_units(monkeypatch):
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=2)
    monkeypatch.setattr(
        Helmholtz2D, "_solve", lambda *a, **k: pytest.fail("fine PDE solve in proposal")
    )
    result = solver.certify_survey(model, blocks, receivers)
    counters = result.costs["certificate"]
    assert (
        counters["high_dst_I_real_transforms"]
        == counters["high_idst_I_real_transforms"]
        == 4 * 2 * len(blocks)
    )
    assert result.costs["batch_dispatches"]["high_dst_I_dispatches"] == 2 * 2 * 2
    assert result.costs["coarse_stability"]["coarse_inverse_rhs"] == 18 * 2
    assert all(value == 0 for value in result.costs["reference_solver"].values())


def test_same_model_content_reuses_reference_and_inplace_model_change_does_not():
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=2)
    first = solver.certify_survey(model, blocks, receivers)
    second = solver.certify_survey(model.copy(), blocks, receivers)
    assert second.costs["certificate"]["high_reference_assemblies"] == 0
    assert np.allclose(first.approximate_gradient, second.approximate_gradient, atol=1e-13)
    model[4] += 0.1
    changed = solver.certify_survey(model, blocks, receivers)
    reference = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=2).certify_survey(
        model, blocks, receivers
    )
    assert changed.model_sha256 != first.model_sha256
    assert changed.costs["certificate"]["high_reference_assemblies"] == 2
    assert np.allclose(
        changed.approximate_gradient, reference.approximate_gradient, rtol=1e-12, atol=1e-12
    )


def test_unavailable_frequency_keeps_every_block_and_requires_recovery():
    p, model, receivers, blocks = case((1.0, 40.0))
    result = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=2).certify_survey(
        model, blocks, receivers
    )
    assert len(result.block_certificates) == len(blocks)
    assert not result.available and not result.certified_descent and result.exact_fallback_required
    assert result.directional_error_bound is result.certified_upper_derivative is None
    unavailable = [b for b in result.block_certificates if not b.stability_available]
    assert len(unavailable) == 2 and all(b.correction_skipped for b in unavailable)
    assert all(np.all(b.state == 0) and np.all(b.adjoint == 0) for b in unavailable)
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_proposal_is_never_exact_stationarity(monkeypatch):
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=0)
    groups = {f: tuple(i for i, b in enumerate(blocks) if b.frequency_hz == f) for f in (1.0, 1.3)}
    monkeypatch.setattr(
        solver,
        "_coarse_fields",
        lambda *a: SimpleNamespace(
            group_indices=groups,
            block_states=np.zeros((p.n, len(blocks)), complex),
            block_adjoints=np.zeros((p.n, len(blocks)), complex),
            failures={},
        ),
    )
    result = solver.certify_survey(model, blocks, receivers)
    assert not result.certified_descent and result.exact_fallback_required
    assert "stationarity" in result.reason and np.all(result.direction == 0)


def test_failed_field_generation_is_reported_not_treated_as_zero_stationarity(monkeypatch):
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=1)
    monkeypatch.setattr(
        solver,
        "_precondition",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("nonfinite synthetic repair")),
    )
    result = solver.certify_survey(model, blocks, receivers)
    assert (
        not result.available
        and result.exact_fallback_required
        and result.approximate_gradient is None
    )
    assert len(result.costs["field_generation_failures"]) == 2
    assert "field generation" in result.reason
    json.dumps(result.to_dict(), allow_nan=False)


def test_finite_field_contraction_overflow_abstains_with_strict_json(monkeypatch):
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=0)
    groups = {f: tuple(i for i, b in enumerate(blocks) if b.frequency_hz == f) for f in (1.0, 1.3)}
    fields = np.full((p.n, len(blocks)), 1e155 + 1e155j)
    monkeypatch.setattr(
        solver,
        "_coarse_fields",
        lambda *a: SimpleNamespace(
            group_indices=groups, block_states=fields, block_adjoints=fields, failures={}
        ),
    )
    result = solver.certify_survey(model, blocks, receivers)
    assert (
        not result.available
        and result.exact_fallback_required
        and result.approximate_gradient is None
    )
    json.dumps(result.to_dict(), allow_nan=False)


def test_result_buffers_are_immutable_and_input_physics_is_snapshotted():
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=1)
    first = solver.certify_survey(model, blocks, receivers)
    p.spacing *= 2
    second = solver.certify_survey(model, blocks, receivers)
    assert np.array_equal(first.approximate_gradient, second.approximate_gradient)
    for array in (first.direction, first.approximate_gradient, first.block_certificates[0].state):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("sweeps", [True, -1, 1.5, 17])
def test_invalid_iteration_parameters_fail(sweeps):
    p, _, _, _ = case()
    with pytest.raises(ValueError):
        DeflatedDirectionCertifier(p, rank=18, repair_sweeps=sweeps)


def test_finite_divergent_repair_and_repeated_receiver_overflow_abstain(monkeypatch):
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=2)
    monkeypatch.setattr(
        solver,
        "_precondition",
        lambda context, rhs, model, **kwargs: np.full(rhs.shape, 1e308 + 0j),
    )
    result = solver.certify_survey(model, blocks, receivers)
    assert not result.available and not result.certified_descent and result.exact_fallback_required
    assert result.approximate_gradient is None
    assert all(
        "nonfinite receiver injection" in value
        for value in result.costs["field_generation_failures"].values()
    )
    json.dumps(result.to_dict(), allow_nan=False)


def test_final_stencil_overflow_is_inside_failed_field_receipt(monkeypatch):
    p, model, receivers, blocks = case()
    solver = DeflatedDirectionCertifier(p, rank=18, repair_sweeps=1)
    monkeypatch.setattr(
        solver,
        "_precondition",
        lambda context, rhs, model, **kwargs: np.full(rhs.shape, 1e307 + 0j),
    )
    result = solver.certify_survey(model, blocks, receivers)
    assert (
        not result.available
        and result.exact_fallback_required
        and result.approximate_gradient is None
    )
    assert len(result.costs["field_generation_failures"]) == 2
    assert result.costs["certificate"]["paired_repair_block_sweep_attempts"] == len(blocks)
    assert all(block.correction_skipped for block in result.block_certificates)
    json.dumps(result.to_dict(), allow_nan=False)
