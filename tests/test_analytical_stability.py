import numpy as np
import pytest

from wavecert.certificates.directional import certify_direction
from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.physics.certified_stability import dirichlet_stability_lower_bound
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.repair.adaptive import adaptive_hybrid_direction, global_fallback_direction
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def _bound(physics, model, frequency, **kwargs):
    return dirichlet_stability_lower_bound(
        shape=physics.shape, spacing=physics.spacing, squared_slowness=model,
        damping=physics.damping, frequency_hz=frequency, **kwargs,
    )


@pytest.mark.parametrize("shape", [(3, 7), (7, 9), (16, 16)])
def test_analytical_bound_never_exceeds_svd_reference(shape):
    rng = np.random.default_rng(230)
    physics = Helmholtz2D(shape=shape)
    for frequency in (0.5, 2.5, 3.0, 4.0, 30.0):
        model = rng.uniform(0.18, 0.25, size=physics.n)
        result = _bound(physics, model, frequency)
        exact = physics.smallest_singular_value(model, frequency)
        assert 0 <= result.beta_lower_bound <= exact + 1e-10


def test_constant_background_bound_handles_indefinite_real_part_without_svd():
    physics = Helmholtz2D(shape=(16, 16))
    model = np.full(physics.n, 0.25)
    result = _bound(physics, model, 3.0)
    real_eigenvalues = np.linalg.eigvalsh(physics.operator(model, 3.0).toarray().real)
    assert real_eigenvalues[0] < 0 < real_eigenvalues[-1]
    assert result.available
    assert result.method == "normal-reference-perturbation"
    assert all(value == 0 for value in physics.cost_snapshot().values())


def test_resonance_without_absorption_returns_unavailable():
    physics = Helmholtz2D(shape=(8, 8), damping_strength=0.0)
    model = np.full(physics.n, 0.25)
    minimum = 8 / physics.spacing**2 * np.sin(np.pi / 18) ** 2
    frequency = np.sqrt(minimum / model[0]) / (2 * np.pi)
    result = _bound(physics, model, frequency)
    assert not result.available
    assert result.beta_lower_bound == 0


def test_analytical_stability_can_drive_valid_directional_certificate():
    problem = build_synthetic_problem(shape=(8, 8), frequencies_hz=(2.5,), n_sources=1)
    physics = problem.physics
    block = problem.blocks[0]
    surrogate = SmoothedHelmholtzSurrogate(physics, smoothing_sigma=0.5)
    q = physics.source(block.source_index)
    _, gradient, *_ = surrogate.objective_and_gradient(
        problem.m0, q, block.observed, problem.geometry.receiver_indices, block.frequency_hz,
    )
    result = _bound(physics, problem.m0, block.frequency_hz)
    assert result.available
    cert = certify_direction(
        physics=physics, surrogate=surrogate, m=problem.m0, q=q, observed=block.observed,
        receiver_indices=problem.geometry.receiver_indices, frequency_hz=block.frequency_hz,
        direction=-gradient / np.linalg.norm(gradient), beta=result.beta_lower_bound,
        stability_mode="analytic", validate_exact=True,
    )
    assert cert.realized_error <= cert.directional_error_bound + 1e-10


@pytest.mark.parametrize("field,value", [("spacing", 0), ("frequency_hz", -1), ("center_points", 1)])
def test_invalid_analytical_bound_inputs_fail(field, value):
    kwargs = dict(
        shape=(4, 4), spacing=0.05, squared_slowness=np.ones(16),
        damping=np.zeros(16), frequency_hz=1.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        dirichlet_stability_lower_bound(**kwargs)


class _LinearSurrogate:
    """An inexpensive differentiable surrogate for fallback accounting tests."""

    def state(self, m, q, frequency_hz):
        return np.asarray(m, dtype=complex)

    def jvp(self, m, direction, q, frequency_hz):
        return np.asarray(direction, dtype=complex)

    def objective_and_gradient(self, m, q, observed, receivers, frequency_hz):
        state = self.state(m, q, frequency_hz)
        residual = state[np.asarray(receivers)] - observed
        gradient = np.zeros(len(m))
        np.add.at(gradient, np.asarray(receivers), residual.real)
        return 0.5 * float(np.vdot(residual, residual).real), gradient, state, residual


class _ForbiddenSurrogate:
    def state(self, *args, **kwargs):
        pytest.fail("unavailable analytic stability must skip neural inference")

    def jvp(self, *args, **kwargs):
        pytest.fail("unavailable analytic stability must skip neural JVPs")

    def objective_and_gradient(self, *args, **kwargs):
        pytest.fail("unavailable analytic stability must skip neural gradients")


@pytest.mark.parametrize("policy", [adaptive_hybrid_direction, global_fallback_direction])
def test_unavailable_analytical_stability_falls_back_without_hidden_svd(monkeypatch, policy):
    problem = build_synthetic_problem(shape=(16, 16), frequencies_hz=(3.0,), n_sources=2)
    physics = problem.physics
    model = np.linspace(0.1, 0.8, physics.n)
    assert not physics.analytical_stability_bound(model, 3.0).available
    physics.reset_cost_counters(clear_cache=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("analytic policy must not use SVD or receiver stability solves")

    monkeypatch.setattr(physics, "smallest_singular_value", unexpected)
    monkeypatch.setattr(physics, "receiver_resolvent_matrix", unexpected)
    result = policy(
        physics=physics, surrogate=_ForbiddenSurrogate(), m=model, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, stability_mode="analytic",
    )
    assert result.mode == "unavailable-stability-exact-fallback"
    assert result.certified_descent
    assert not result.surrogate_gradient_evaluated
    assert result.surrogate_gradient_norm == 0
    assert result.exact_block_evaluations == len(problem.blocks)
    assert result.exact_objective is not None
    cost = physics.cost_snapshot()
    assert cost["forward_rhs"] == len(problem.blocks)
    assert cost["adjoint_rhs"] == len(problem.blocks)
    assert cost["factorizations"] == 1
    assert cost["analytic_bound_evaluations"] == 1
    assert cost["svd_evaluations"] == cost["stability_rhs"] == 0
    reference_value = sum(
        physics.objective(model, physics.source(block.source_index), block.observed,
                          problem.geometry.receiver_indices, block.frequency_hz)
        for block in problem.blocks
    )
    assert result.exact_objective == pytest.approx(reference_value)


def test_analytical_unavailability_respects_exact_block_budget():
    problem = build_synthetic_problem(shape=(16, 16), frequencies_hz=(3.0,), n_sources=2)
    model = np.linspace(0.1, 0.8, problem.physics.n)
    result = adaptive_hybrid_direction(
        physics=problem.physics, surrogate=_ForbiddenSurrogate(), m=model, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, stability_mode="analytic", max_exact_blocks=1,
    )
    assert not result.certified_descent
    assert result.mode == "unavailable-stability-budget-exhausted"
    assert result.exact_block_evaluations == 0
    assert not result.surrogate_gradient_evaluated
    assert np.all(result.direction == 0)
    with pytest.raises(ValueError, match="analytical stability bound unavailable"):
        certify_direction(
            physics=problem.physics, surrogate=_ForbiddenSurrogate(), m=model,
            q=problem.physics.source(problem.blocks[0].source_index),
            observed=problem.blocks[0].observed, receiver_indices=problem.geometry.receiver_indices,
            frequency_hz=3.0, direction=np.ones(problem.physics.n), stability_mode="analytic",
        )


def test_available_analytical_preflight_is_reused_without_extra_bound_computations():
    problem = build_synthetic_problem(shape=(8, 8), frequencies_hz=(2.5,), n_sources=2)
    physics = problem.physics
    assert physics.analytical_stability_bound(problem.m0, 2.5).available
    physics.reset_cost_counters(clear_cache=True)
    result = adaptive_hybrid_direction(
        physics=physics, surrogate=_LinearSurrogate(), m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, stability_mode="analytic", max_exact_blocks=0,
    )
    assert result.surrogate_gradient_evaluated
    assert physics.cost_snapshot()["analytic_bound_evaluations"] == 1
    assert physics.cost_snapshot()["svd_evaluations"] == 0


@pytest.mark.parametrize("policy", [adaptive_hybrid_direction, global_fallback_direction])
def test_full_repair_retains_online_objective_for_reuse(monkeypatch, policy):
    from types import SimpleNamespace

    problem = build_synthetic_problem(shape=(8, 8), frequencies_hz=(2.5,), n_sources=2)
    # Force this unit test through the full-repair branch. This tests retained
    # online values, not the sharpness or validity of a replacement bound.
    monkeypatch.setattr(
        "wavecert.repair.adaptive.certify_direction",
        lambda **kwargs: SimpleNamespace(directional_error_bound=1e12, surrogate_directional_derivative=0.0),
    )
    result = policy(
        physics=problem.physics, surrogate=_LinearSurrogate(), m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, stability_mode="analytic",
    )
    reference = sum(
        problem.physics.objective(
            problem.m0, problem.physics.source(block.source_index), block.observed,
            problem.geometry.receiver_indices, block.frequency_hz,
        )
        for block in problem.blocks
    )
    assert result.exact_block_evaluations == len(problem.blocks)
    assert result.exact_objective == pytest.approx(reference)
