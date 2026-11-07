"""Independent explicit small-grid checks; dense matrices occur only in tests."""

import json

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D


def explicit_basis(engine):
    nz, nx = engine.physics_config["shape"]
    z, x = np.arange(1, nz + 1), np.arange(1, nx + 1)
    return np.stack([
        (2 / np.sqrt((nz + 1) * (nx + 1))
         * np.sin(np.pi * z[:, None] * k / (nz + 1))
         * np.sin(np.pi * x[None, :] * p / (nx + 1))).reshape(-1)
        for k, p in engine.mode_indices], axis=1)


@pytest.mark.parametrize("shape", [(5, 6), (6, 5)])
def test_fast_moments_projection_and_high_frequency_folding(shape):
    rng = np.random.default_rng(122)
    p = Helmholtz2D(shape=shape, spacing=0.05, damping_strength=2.3)
    engine = FastCoarseStability(p, p.n - 2)
    m = rng.uniform(0.1, 0.3, p.n)
    result = engine.evaluate(m, 3.0)
    assert result.available, result.reason
    b = explicit_basis(engine)
    np.testing.assert_allclose(b.T @ b, np.eye(engine.rank), atol=3e-15)
    a = p.operator(m, 3.0).toarray()
    ar = b.T @ a @ b
    coupling = a @ b - b @ ar
    np.testing.assert_allclose(result.reduced_operator, ar, atol=3e-12, rtol=3e-13)
    np.testing.assert_allclose(result.raw_coupling_gram,
                               coupling.conjugate().T @ coupling, atol=3e-10, rtol=2e-11)
    # Highest selected mode sums exceed the DCT-I Nyquist index on both axes.
    assert np.any(2 * engine.mode_indices[:, 0] > shape[0] + 1)
    assert np.any(2 * engine.mode_indices[:, 1] > shape[1] + 1)
    field = rng.normal(size=p.n) + 1j * rng.normal(size=p.n)
    coeff = result.project(field)
    np.testing.assert_allclose(coeff, b.T @ field, atol=3e-15)
    np.testing.assert_allclose(result.lift(coeff), b @ coeff, atol=3e-15)
    np.testing.assert_allclose(result.component_norms(field),
                               [la.norm(b.T @ field), la.norm(field - b @ (b.T @ field))])
    np.testing.assert_allclose(result.apply_operator(field), a @ field, atol=2e-12)
    np.testing.assert_allclose(result.apply_operator(field, adjoint=True),
                               a.conjugate().T @ field, atol=2e-12)


def test_component_inverse_bounds_and_complex_adjoint_against_independent_fine_solves():
    rng = np.random.default_rng(891)
    p = Helmholtz2D(shape=(6, 7), spacing=0.1, damping_strength=1.7)
    engine = FastCoarseStability(p, 15)
    m = rng.uniform(0.1, 0.35, p.n)
    result = engine.evaluate(m, 2.0)
    assert result.available, result.reason
    a = p.operator(m, 2.0)
    b = explicit_basis(engine)
    ar = b.T @ (a @ b)
    coupling = a @ b - b @ ar
    exact_k = la.norm(la.solve(ar, coupling.T), 2)
    # Conjugating G is essential for this complex-symmetric block identity.
    assert result.k >= exact_k * (1 - 1e-11)
    assert result.k <= exact_k * (1 + 1e-6)
    assert result.beta_lower_bound <= la.svdvals(a.toarray())[-1] * (1 + 1e-11)
    residual = rng.normal(size=p.n) + 1j * rng.normal(size=p.n)
    for adjoint in (False, True):
        fine = a.conjugate().T if adjoint else a
        error = spla.spsolve(fine.tocsc(), residual)
        components = result.component_norms(error)
        generic = result.M @ result.component_norms(residual)
        oriented = result.oriented_error_bounds(residual, adjoint=adjoint)
        assert np.all(components <= generic * (1 + 1e-11))
        assert np.all(components <= oriented * (1 + 1e-11))
        assert np.all(oriented <= generic * (1 + 1e-11))
        np.testing.assert_allclose(result.solve_coarse(b.T @ residual, adjoint=adjoint),
            la.solve(ar.conjugate().T if adjoint else ar, b.T @ residual), rtol=1e-12)
    cost = engine.cost_snapshot()
    assert cost["coarse_inverse_rhs"] == engine.rank
    assert cost["coarse_oriented_rhs"] == 2
    assert cost["fine_grid_factorizations"] == cost["fine_grid_rhs"] == 0
    assert cost["fine_grid_dense_svd"] == 0


def test_corrections_and_physical_adjoint_error_with_repeated_receivers():
    rng = np.random.default_rng(94)
    p = Helmholtz2D(shape=(5, 6), spacing=0.08)
    engine = FastCoarseStability(p, 12)
    m = rng.uniform(0.13, 0.25, p.n)
    result = engine.evaluate(m, 2.0)
    assert result.available
    a = p.operator(m, 2.0).tocsc()
    b = explicit_basis(engine)
    idx = (2, 2, 7, 19, 19)
    pmat = np.eye(p.n)[list(idx)]
    constants = result.receiver_constants(idx)
    np.testing.assert_allclose(constants["basis"], pmat @ b, atol=2e-15)
    assert constants["pB"] >= la.norm(pmat @ b, 2)
    assert constants["pQ"] >= la.norm(pmat @ (np.eye(p.n) - b @ b.T), 2)
    np.testing.assert_allclose(constants["rho_adjoint"],
        la.norm(la.solve(result.reduced_operator.conjugate().T, (pmat @ b).T), 2))
    receiver_rhs = engine.cost_snapshot()["coarse_receiver_rhs"]
    assert receiver_rhs == len(idx)
    constants["pQ"] = 12345  # Caller metadata mutation cannot poison the cache.
    assert result.receiver_constants(idx)["pQ"] < 2
    assert engine.cost_snapshot()["coarse_receiver_rhs"] == receiver_rhs
    q = rng.normal(size=p.n) + 1j * rng.normal(size=p.n)
    u = spla.spsolve(a, q)
    uh, rp = result.correct(np.zeros(p.n, dtype=complex), q)
    np.testing.assert_allclose(result.project(rp), 0, atol=1e-13)
    np.testing.assert_allclose(rp, a @ uh - q, atol=2e-14)
    eta_u = result.oriented_error_bounds(rp)
    assert np.all(result.component_norms(u - uh) <= eta_u * (1 + 1e-11))
    eta_pu = constants["pB"] * eta_u[0] + result.receiver_constants(idx)["pQ"] * eta_u[1]
    # Use the unmodified pB and the cached pQ, recomputing the physical adjoint rhs.
    observed = rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))
    exact_rhs = pmat.T @ (pmat @ u - observed)
    approx_rhs = pmat.T @ (pmat @ uh - observed)
    exact_lambda = spla.spsolve(a.conjugate().T, exact_rhs)
    lh, ra = result.correct(np.zeros(p.n, dtype=complex), approx_rhs, adjoint=True)
    np.testing.assert_allclose(result.project(ra), 0, atol=1e-13)
    eta_lambda = result.oriented_adjoint_error_bounds(ra, eta_pu, idx)
    assert np.all(result.component_norms(exact_lambda - lh) <= eta_lambda * (1 + 1e-11))
    # R > rank takes an explicitly conservative path without forming N by r.
    many = result.receiver_constants(tuple(range(p.n)) + (1, 1, 1))
    assert many["basis"] is None
    assert many["pB"] == many["pQ"] == 2.0
    assert "conservative" in many["mode"]


def test_constant_gram_cancellation_near_resonance_and_singular_fail_closed():
    p = Helmholtz2D(shape=(5, 6), spacing=0.1, damping_strength=0)
    engine = FastCoarseStability(p, 12)
    frequency = 2.0
    omega2 = (2 * np.pi * frequency)**2
    m = np.full(p.n, (engine.laplacian_eigenvalues[0] + 0.01) / omega2)
    result = engine.evaluate(m, frequency)
    assert result.available, result.reason
    assert la.norm(result.raw_coupling_gram) < 1e-9
    assert result.numerical_diagnostics["gram_diagonal_shift"] > 0
    assert not result.numerical_diagnostics["formally_verified_roundoff_enclosure"]
    assert result.beta_lower_bound <= 0.01 * (1 + 1e-7)
    singular_m = np.full(p.n, engine.laplacian_eigenvalues[0] / omega2)
    singular = engine.evaluate(singular_m, frequency)
    assert not singular.available
    assert "singular" in singular.reason
    with pytest.raises(ValueError, match="unavailable"):
        singular.solve_coarse(np.ones(engine.rank))
    # Projection and actual stencil evaluation remain usable for reporting.
    assert singular.project(np.ones(p.n)).shape == (engine.rank,)
    assert singular.apply_operator(np.ones(p.n)).shape == (p.n,)
    json.dumps(singular.to_dict(), allow_nan=False)


def test_snapshot_immutability_model_byte_cache_and_no_fine_solver_calls(monkeypatch):
    p = Helmholtz2D(shape=(5, 6), spacing=0.1)
    engine = FastCoarseStability(p, 10, model_cache_size=1)
    m = np.full(p.n, 0.2)
    first = engine.evaluate(m, 2.0)
    first_model_hash = first.to_dict()["model_sha256"]
    with pytest.raises(AttributeError):
        first.frequency_hz = 123.0
    with pytest.raises(AttributeError):
        first.omega = 123.0
    reference = first.apply_operator(np.ones(p.n))
    signature = engine.operator_signature
    cfg = engine.physics_config
    cfg["shape"][0] = 123
    cfg["spacing"] = 123
    p.shape, p.spacing, p.damping_strength = (3, 4), 3, 123
    m[:] = 0.21
    np.testing.assert_array_equal(first.apply_operator(np.ones(engine.n)), reference)
    assert engine.operator_signature == signature
    assert engine.physics_config["shape"] == [5, 6]

    def prohibited(*args, **kwargs):
        raise AssertionError("fine solver/matrix construction is prohibited")
    monkeypatch.setattr(Helmholtz2D, "operator", prohibited)
    monkeypatch.setattr(Helmholtz2D, "solve_state", prohibited)
    engine.evaluate(np.full(engine.n, 0.2), 2.1)
    assert engine.cost_snapshot()["model_cache_hits"] == 1
    changed = engine.evaluate(m, 2.0)
    assert changed.to_dict()["model_sha256"] != first_model_hash
    assert first.to_dict()["model_sha256"] == first_model_hash
    assert engine.cost_snapshot()["online_dct_I_real_transforms"] == 4
    m[:] = 0.2
    engine.evaluate(m, 2.0)
    assert engine.cost_snapshot()["online_dct_I_real_transforms"] == 6  # bounded eviction
    assert engine.cost_snapshot()["offline_dct_I_real_transforms"] == 2
    assert all(isinstance(v, int) for v in engine.cost_snapshot().values())
    engine.reset_cost_counters(clear_cache=True)
    assert all(v == 0 for v in engine.cost_snapshot().values())


def test_explicit_unavailability_and_validation(monkeypatch):
    p = Helmholtz2D(shape=(5, 6), spacing=0.1)
    engine = FastCoarseStability(p, 1)
    assert "gamma" in engine.evaluate(np.ones(p.n), 100).reason
    assert "nonfinite" in engine.evaluate(np.ones(p.n), 1e308).reason
    # A real heterogeneous model can fail small gain even with gamma > 0.
    m = np.linspace(0.01, 0.6, p.n)
    small_gain = engine.evaluate(m, 1.8)
    assert small_gain.gamma > 0
    assert not small_gain.available and "tau" in small_gain.reason
    engine = FastCoarseStability(p, 8)
    original = engine._multiplication
    calls = 0

    def corrupted_second_moment(field, **kwargs):
        nonlocal calls
        calls += 1
        return original(field, **kwargs) if calls == 1 else np.zeros((engine.rank, engine.rank))
    monkeypatch.setattr(engine, "_multiplication", corrupted_second_moment)
    bad = engine.evaluate(np.full(p.n, 0.2), 2.0)
    assert not bad.available and "strongly indefinite" in bad.reason
    for invalid in (np.zeros(p.n), np.full(p.n, np.nan), np.ones(p.n - 1)):
        with pytest.raises(ValueError):
            engine.evaluate(invalid, 2.0)
    for imaginary in (0.0, 0.1):
        with pytest.raises(ValueError, match="must be real"):
            engine.evaluate(np.full(p.n, 0.2 + 1j * imaginary), 2.0)
    for rank in (True, 0, p.n):
        with pytest.raises(ValueError):
            FastCoarseStability(p, rank)
    with pytest.raises(ValueError):
        FastCoarseStability(p, 2, model_cache_size=-1)
    result = FastCoarseStability(p, 8).evaluate(np.full(p.n, 0.2), 2.0)
    for indices in ((-1,), (p.n,), (1.2,), (True,)):
        with pytest.raises(ValueError):
            result.receiver_constants(indices)
    for invalid in (-1, np.nan):
        with pytest.raises(ValueError):
            result.oriented_adjoint_error_bounds(np.ones(p.n), invalid, (1,))
