"""Synthetic checks only.  No project model/checkpoint/test arrays are loaded."""
import json

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wavecert.physics.coarse_dst_reference import (
    CoarseDSTEngine,
    LinearSolveFailure,
    PreparationUnavailable,
)
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D


def dense_operator(shape, spacing, model, damping, frequency):
    """Independent Kronecker construction; the solver uses only a stencil."""
    nz, nx = shape
    tx = sp.diags([-np.ones(nx-1), 2*np.ones(nx), -np.ones(nx-1)], [-1, 0, 1])
    tz = sp.diags([-np.ones(nz-1), 2*np.ones(nz), -np.ones(nz-1)], [-1, 0, 1])
    stiffness = (sp.kron(sp.eye(nz), tx)+sp.kron(tz, sp.eye(nx)))/spacing**2
    omega = 2*np.pi*frequency
    return (stiffness+sp.diags(-omega**2*model+1j*omega*damping)).tocsc()


def sine_basis(shape, modes):
    nz, nx = shape
    z, x = np.arange(1, nz+1)[:, None], np.arange(1, nx+1)[None, :]
    return np.column_stack([(2/np.sqrt((nz+1)*(nx+1))*np.sin(np.pi*k*z/(nz+1))
                             * np.sin(np.pi*p*x/(nx+1))).ravel() for k, p in modes])


@pytest.fixture
def problem():
    shape, spacing, frequency = (5, 7), .1, 2.5
    model = np.linspace(.16, .3, np.prod(shape))
    damping = .3+np.linspace(0, .5, len(model))**2
    factory = CoarseDSTEngine(shape, spacing, damping, 8)
    return factory, model, damping, frequency


@pytest.mark.parametrize("adjoint", [False, True])
def test_nonsquare_operator_galerkin_inverse_and_complex_multi_rhs_lu(problem, adjoint):
    engine, model, damping, frequency = problem
    context = engine.prepare(model, frequency)
    rng = np.random.default_rng(31)
    rhs = rng.normal(size=(engine.n, 3))+1j*rng.normal(size=(engine.n, 3))
    matrix = dense_operator(engine.shape, .1, model, damping, frequency)
    basis = sine_basis(engine.shape, engine.mode_indices)
    np.testing.assert_allclose(context.reduced_operator, basis.T@matrix@basis, atol=3e-13)
    operator = matrix.conjugate().T if adjoint else matrix
    np.testing.assert_allclose(context.apply_operator(rhs, adjoint=adjoint), operator@rhs, atol=4e-13)
    answer = context.solve_report(rhs, adjoint=adjoint)
    expected = spla.splu(matrix).solve(rhs, trans="H" if adjoint else "N")
    assert answer.converged
    np.testing.assert_allclose(answer.solution, expected, rtol=2e-9, atol=2e-11)
    for j, report in enumerate(answer.columns):
        true = la.norm(operator@answer.solution[:, j]-rhs[:, j])
        assert true <= max(1e-12, 1e-10*la.norm(rhs[:, j]))
        assert report.true_residual_norm == pytest.approx(true, abs=5e-14)
    assert answer.costs["linear_solve_columns"] == 3
    assert answer.costs["arnoldi_iterations"] == sum(c.iterations for c in answer.columns)
    json.dumps(answer.to_dict(), allow_nan=False)


@pytest.mark.parametrize("adjoint", [False, True])
def test_same_preconditioner_action_as_existing_corrected_high_inverse(adjoint):
    physics = Helmholtz2D(shape=(6, 8), spacing=.1, damping_width=2, damping_strength=.8)
    model = np.linspace(.12, .2, physics.n)
    engine = CoarseDSTEngine.from_helmholtz(physics, 14)
    context = engine.prepare(model, 1.7)
    certificate = FastCoarseStability(physics, 14).evaluate(model, 1.7)
    assert certificate.available
    rng = np.random.default_rng(12)
    rhs = rng.normal(size=(physics.n, 2))+1j*rng.normal(size=(physics.n, 2))
    actual = context.apply_preconditioner(rhs, adjoint=adjoint)
    for j in range(2):
        high = context.high_inverse(rhs[:, j], adjoint=adjoint)
        expected, _ = certificate.correct(high, rhs[:, j], adjoint=adjoint)
        np.testing.assert_allclose(actual[:, j], expected, rtol=2e-13, atol=3e-15)
        remaining = context.project(rhs[:, j]-context.apply_operator(actual[:, j], adjoint=adjoint))
        assert la.norm(remaining) < 5e-13*la.norm(rhs[:, j])


def test_preparation_has_no_certificate_la_and_reuses_one_lu_for_adjoint(problem, monkeypatch):
    engine, model, _, frequency = problem
    def forbidden(*args, **kwargs):
        raise AssertionError("certificate SVD/eigen work must not enter exact baseline")
    monkeypatch.setattr(la, "svdvals", forbidden)
    monkeypatch.setattr(la, "eigvalsh", forbidden)
    context = engine.prepare(model, frequency)
    rhs = np.ones(engine.n)+1j*np.arange(engine.n)/engine.n
    assert context.solve_report(rhs).converged
    assert context.solve_report(rhs, adjoint=True).converged
    assert engine.prepare(model.copy(), frequency) is context
    engine.prepare(model, frequency+.25)
    counts = engine.cost_snapshot()
    assert counts["offline_dct_I_real_transforms"] == 1
    assert counts["online_dct_I_real_transforms"] == 1
    assert counts["coarse_lu_factorizations"] == 2
    assert counts["context_cache_hits"] == 1
    assert counts["model_moment_cache_hits"] == 1
    assert counts["coarse_forward_rhs"] > 0 and counts["coarse_adjoint_rhs"] > 0
    assert all(counts[key] == 0 for key in ("fine_factorizations", "fine_direct_rhs",
               "certificate_svd_evaluations", "certificate_eigenproblems", "certificate_inverse_rhs"))
    assert all(type(value) is int and value >= 0 for value in counts.values())


def test_paid_matrix_preconditioner_columns_dispatches_and_gmres_work_invariant(problem):
    engine, model, _, frequency = problem
    context = engine.prepare(model, frequency)
    before = context.cost_snapshot()
    context.apply_preconditioner(np.ones((engine.n, 2)), adjoint=True)
    after = context.cost_snapshot()
    delta = {key: after[key]-before[key] for key in after}
    assert delta["preconditioner_dispatches"] == 1 and delta["preconditioner_columns"] == 2
    assert delta["operator_dispatches"] == 1 and delta["adjoint_operator_columns"] == 2
    assert delta["coarse_solve_dispatches"] == 1 and delta["coarse_adjoint_rhs"] == 2
    assert delta["high_inverse_columns"] == delta["projection_columns"] == delta["lift_columns"] == 2
    assert delta["dst_I_dispatches"] == delta["idst_I_dispatches"] == 2
    assert delta["dst_I_real_column_transforms"] == delta["idst_I_real_column_transforms"] == 8
    answer = context.solve_report(np.arange(engine.n)+1j)
    assert answer.converged
    work = answer.costs
    assert work["preconditioner_columns"] == work["arnoldi_iterations"]
    assert work["operator_columns"] == 2*work["arnoldi_iterations"]+work["true_residual_checks"]
    assert work["dst_I_dispatches"] == work["idst_I_dispatches"] == 2*work["arnoldi_iterations"]


def test_immutable_content_cache_and_explicit_real_damping(problem):
    engine, model, damping, frequency = problem
    original_model, original_damping = model.copy(), damping.copy()
    context = engine.prepare(model, frequency)
    old_identity, old_action = context.identity, context.apply_operator(np.ones(engine.n))
    model[0] *= 1.5
    damping[:] = 99
    assert context.identity == old_identity
    np.testing.assert_array_equal(context.model, original_model)
    np.testing.assert_array_equal(context.apply_operator(np.ones(engine.n)), old_action)
    assert engine.prepare(model, frequency).identity["context_sha256"] != old_identity["context_sha256"]
    assert engine.prepare(original_model, frequency) is context
    assert engine.signature["damping_float64_le_sha256"] != CoarseDSTEngine(
        engine.shape, .1, damping, 8).signature["damping_float64_le_sha256"]
    receipt = context.identity
    receipt["rank"] = 999
    assert context.identity["rank"] == 8
    for target, name in ((engine, "rank"), (context, "frequency_hz"), (context, "omega")):
        with pytest.raises(AttributeError):
            setattr(target, name, 1)
    with pytest.raises(ValueError, match="real"):
        CoarseDSTEngine(engine.shape, .1, original_damping.astype(complex), 8)
    with pytest.raises(ValueError, match="real"):
        engine.prepare(original_model.astype(complex), frequency)


def test_scalar_matrix_and_exact_warm_start_columns(problem):
    engine, model, damping, frequency = problem
    context = engine.prepare(model, frequency)
    rng = np.random.default_rng(2)
    rhs = rng.normal(size=(engine.n, 2))
    matrix = dense_operator(engine.shape, .1, model, damping, frequency)
    warm = spla.splu(matrix).solve(rhs.astype(complex))
    answer = context.solve_report(rhs, x0=warm)
    assert answer.converged and all(c.iterations == 0 for c in answer.columns)
    assert answer.costs["preconditioner_columns"] == 0
    np.testing.assert_array_equal(answer.solution, warm)
    single = context.solve(rhs[:, 0])
    assert single.shape == (engine.n,)
    np.testing.assert_allclose(single, warm[:, 0], atol=1e-10)
    zero = context.solve_report(np.zeros(engine.n))
    assert zero.converged and zero.columns[0].iterations == 0
    with pytest.raises(ValueError, match="same shape"):
        context.solve_report(rhs, x0=warm[:, 0])
    with pytest.raises(ValueError, match="shape"):
        context.solve_report(np.empty((engine.n, 0)))


@pytest.mark.parametrize("repeated_sampling", [False, True])
def test_physical_multi_source_gradient_fd_arbitrary_complex_or_repeated_P(problem, repeated_sampling):
    engine, model, damping, frequency = problem
    rng = np.random.default_rng(47)
    rhs = rng.normal(size=(engine.n, 3))+1j*rng.normal(size=(engine.n, 3))
    if repeated_sampling:
        sampling = np.eye(engine.n)[[2, 5, 5, 12, 18, 18, 20]]
    else:
        sampling = (rng.normal(size=(7, engine.n))+1j*rng.normal(size=(7, engine.n)))/engine.n
    observed = rng.normal(size=(7, 3))+1j*rng.normal(size=(7, 3))
    context = engine.prepare(model, frequency)
    state = context.solve(rhs)
    residual = sampling@state-observed
    adjoint = context.solve(sampling.conjugate().T@residual, adjoint=True)
    gradient = context.omega**2*np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
    exact_factor = spla.splu(dense_operator(engine.shape, .1, model, damping, frequency))
    exact_state = exact_factor.solve(rhs)
    exact_adjoint = exact_factor.solve(sampling.conjugate().T@(sampling@exact_state-observed), trans="H")
    expected = context.omega**2*np.real(np.sum(np.conjugate(exact_adjoint)*exact_state, axis=1))
    np.testing.assert_allclose(gradient, expected, rtol=2e-8, atol=2e-10)
    direction = rng.normal(size=engine.n)
    direction /= la.norm(direction)
    def objective(m):
        value = sampling@engine.prepare(m, frequency).solve(rhs)-observed
        return .5*float(np.vdot(value, value).real)
    errors = [abs((objective(model+h*direction)-objective(model-h*direction))/(2*h)
                  - gradient@direction) for h in (1e-3, 3e-4, 1e-4)]
    assert min(errors) <= 3e-7*max(1., abs(gradient@direction))


def test_right_gmres_contains_same_sweep_iterate_at_fixed_work(problem):
    engine, model, _, frequency = problem
    context = engine.prepare(model, frequency)
    rng = np.random.default_rng(7)
    rhs = rng.normal(size=engine.n)+1j*rng.normal(size=engine.n)
    initial = context.lift(context.solve_coarse(context.project(rhs)))
    richardson = initial.copy()
    for _ in range(3):
        richardson += context.apply_preconditioner(rhs-context.apply_operator(richardson))
    gmres = context.solve_report(rhs, x0=initial, rtol=0., atol=0., maxiter=3)
    assert gmres.columns[0].iterations == 3
    assert la.norm(rhs-context.apply_operator(gmres.solution)) <= (
        la.norm(rhs-context.apply_operator(richardson))*(1+1e-9)+1e-13)


def test_iteration_failure_restarts_and_true_residual_floor(problem):
    engine, model, _, frequency = problem
    context = engine.prepare(model, frequency)
    rhs = np.arange(engine.n)+1j
    failed = context.solve_report(rhs, maxiter=1)
    assert not failed.converged and failed.columns[0].reason == "iteration_limit"
    actual = la.norm(rhs-context.apply_operator(failed.solution))
    assert failed.columns[0].true_residual_norm == pytest.approx(actual, abs=2e-13)
    with pytest.raises(LinearSolveFailure) as caught:
        context.solve(rhs, maxiter=1)
    assert not caught.value.result.converged
    restarted = context.solve_report(rhs, restart=3, maxiter=60)
    assert restarted.converged and restarted.columns[0].restart_cycles > 1
    # An algebraically exact constant-potential inverse still has roundoff.
    uniform = CoarseDSTEngine((4, 6), .1, np.full(24, .3), 6).prepare(np.full(24, .2), 2.)
    strict = uniform.solve_report(np.arange(24)+1j, rtol=0., atol=1e-30)
    assert not strict.converged
    assert strict.columns[0].true_residual_norm > strict.columns[0].tolerance
    assert strict.columns[0].reason == "arnoldi_breakdown_before_true_convergence"


def test_invalid_and_nonfinite_failures_are_explicit_and_json_finite(problem, monkeypatch):
    engine, model, _, frequency = problem
    context = engine.prepare(model, frequency)
    with pytest.raises(ValueError, match="finite"):
        context.solve(np.full(engine.n, np.nan))
    with pytest.raises(ValueError, match="positive"):
        engine.prepare(-model, frequency)
    with pytest.raises(ValueError, match="finite"):
        engine.prepare(model, np.inf)
    with pytest.raises(PreparationUnavailable):
        engine.prepare(model, 1e200)
    with pytest.raises(ValueError, match="stiffness"):
        CoarseDSTEngine((3, 4), 1e-300, np.ones(12), 3)
    bad = context.solve_report(np.ones(engine.n), x0=np.full(engine.n, 1e308))
    assert not bad.converged and bad.columns[0].reason.startswith("numerical_failure")
    json.dumps(bad.to_dict(), allow_nan=False)
    monkeypatch.setattr(context, "apply_preconditioner", lambda *args, **kwargs: np.full(engine.n, np.nan))
    corrupt = context.solve_report(np.ones(engine.n))
    assert not corrupt.converged
    assert "ValueError" in corrupt.columns[0].reason
    json.dumps(corrupt.to_dict(), allow_nan=False)


def test_from_helmholtz_snapshots_current_config_instead_of_stale_cached_damping():
    physics = Helmholtz2D(shape=(4, 6), spacing=.1, damping_strength=.3)
    _ = physics.damping
    physics.damping_strength = 1.2
    engine = CoarseDSTEngine.from_helmholtz(physics, 6)
    fresh = Helmholtz2D(shape=(4, 6), spacing=.1, damping_strength=1.2)
    context = engine.prepare(np.full(24, .2), 2.)
    np.testing.assert_allclose(context.apply_operator(np.ones(24)), fresh.operator(np.full(24, .2), 2.)@np.ones(24))


def test_large_grid_default_bounds_krylov_storage_and_retains_true_stopping(monkeypatch):
    shape, n = (8, 9), 72
    context = CoarseDSTEngine(shape, .1, np.linspace(.2, .6, n), 12).prepare(np.linspace(.16, .3, n), 2.5)
    rhs = np.random.default_rng(2026).normal(size=n)+1j*np.arange(n)/n
    original_zeros = np.zeros
    allocated = []

    def bounded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == n:
            allocated.append(shape)
            assert shape[1] <= 51, "default must not allocate an n-wide Krylov basis"
        return original_zeros(shape, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(np, "zeros", bounded_zeros)
        default = context.solve_report(rhs)
    assert allocated and default.restart == 50 and default.converged
    assert la.norm(rhs-context.apply_operator(default.solution)) <= max(1e-12, 1e-10*la.norm(rhs))
    explicit = context.solve_report(rhs, restart=n)
    assert explicit.restart == n and explicit.converged
    assert la.norm(rhs-context.apply_operator(explicit.solution)) <= max(1e-12, 1e-10*la.norm(rhs))
    capped = context.solve_report(rhs, maxiter=1)
    assert capped.restart == 50 and not capped.converged
    assert capped.columns[0].reason == "iteration_limit"
    uniform = CoarseDSTEngine(shape, .1, np.full(n, .3), 12).prepare(np.full(n, .2), 2.)
    strict = uniform.solve_report(rhs, rtol=0., atol=1e-30)
    assert strict.restart == 50 and not strict.converged
    assert strict.columns[0].true_residual_norm > strict.columns[0].tolerance
    assert strict.columns[0].reason == "arnoldi_breakdown_before_true_convergence"
    json.dumps(strict.to_dict(), allow_nan=False)
