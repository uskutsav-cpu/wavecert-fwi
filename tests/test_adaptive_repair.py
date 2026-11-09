import numpy as np
import pytest

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.repair.adaptive import adaptive_hybrid_direction, global_fallback_direction
from wavecert.statistics.pathwise import pathwise_case_score_details
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_rigorous_adaptive_repair_returns_true_descent_direction():
    problem = build_synthetic_problem(
        shape=(8, 8), frequencies_hz=(3.0, 3.5), n_sources=2, n_receivers=6
    )
    surrogate = SmoothedHelmholtzSurrogate(
        problem.physics, smoothing_sigma=0.8, model_scale=1.015
    )
    result = adaptive_hybrid_direction(
        physics=problem.physics,
        surrogate=surrogate,
        m=problem.m0,
        blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices,
        stability_mode="directional",
    )
    exact_gradient = np.zeros(problem.physics.n)
    for block in problem.blocks:
        q = problem.physics.source(block.source_index)
        _, g, _, _ = problem.physics.objective_and_gradient(
            problem.m0,
            q,
            block.observed,
            problem.geometry.receiver_indices,
            block.frequency_hz,
        )
        exact_gradient += g
    assert result.certified_descent
    assert float(np.dot(exact_gradient, result.direction)) < 0.0
    assert result.exact_block_evaluations <= len(problem.blocks)


class _UnusableGradientSurrogate:
    """A dead network, or a network whose VJP overflows."""

    def __init__(self, invalid=False):
        self.invalid = invalid

    def state(self, m, q, frequency_hz):
        return np.zeros_like(q, dtype=complex)

    def jvp(self, m, direction, q, frequency_hz):
        return np.zeros_like(q, dtype=complex)

    def objective_and_gradient(self, m, q, observed, receivers, frequency_hz):
        gradient = np.zeros_like(m)
        if self.invalid:
            gradient[0] = np.nan
        return (
            0.5 * float(np.vdot(observed, observed).real), gradient,
            self.state(m, q, frequency_hz), -observed,
        )


def _small_problem():
    return build_synthetic_problem(
        shape=(8, 8), frequencies_hz=(3.0,), n_sources=2, n_receivers=6,
    )


def _exact_total(problem):
    return np.sum([
        problem.physics.objective_and_gradient(
            problem.m0, problem.physics.source(block.source_index), block.observed,
            problem.geometry.receiver_indices, block.frequency_hz,
        )[1]
        for block in problem.blocks
    ], axis=0)


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("method", [adaptive_hybrid_direction, global_fallback_direction])
def test_unusable_neural_gradient_recovers_true_descent_with_full_budget(method, invalid):
    problem = _small_problem()
    exact_gradient = _exact_total(problem)
    assert np.linalg.norm(exact_gradient) > 0.1
    result = method(
        physics=problem.physics, surrogate=_UnusableGradientSurrogate(invalid),
        m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices,
    )
    assert result.certified_descent
    assert result.mode == (
        "invalid-hybrid-gradient-exact-fallback" if invalid else
        "zero-hybrid-gradient-exact-fallback"
    )
    assert result.exact_block_evaluations == len(problem.blocks)
    assert float(np.dot(exact_gradient, result.direction)) < 0
    assert result.surrogate_gradient_evaluated
    assert result.surrogate_gradient_valid is not invalid
    assert np.isfinite(result.surrogate_gradient_norm)
    assert np.isfinite(result.final_hybrid_gradient_norm)
    assert np.isfinite(result.exact_objective)


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("budget", [0, 1])
def test_unusable_neural_gradient_abstains_without_exceeding_exact_budget(invalid, budget):
    problem = _small_problem()
    problem.physics.reset_cost_counters(clear_cache=True)
    result = adaptive_hybrid_direction(
        physics=problem.physics, surrogate=_UnusableGradientSurrogate(invalid),
        m=problem.m0, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, max_exact_blocks=budget,
    )
    assert not result.certified_descent
    assert result.mode == (
        "invalid-hybrid-gradient-budget-exhausted" if invalid else
        "zero-hybrid-gradient-budget-exhausted"
    )
    assert result.exact_block_evaluations == 0
    assert problem.physics.cost_snapshot()["adjoint_rhs"] == 0
    np.testing.assert_array_equal(result.direction, np.zeros_like(problem.m0))
    assert np.isfinite(result.surrogate_gradient_norm)
    assert np.isfinite(result.final_hybrid_gradient_norm)


def test_zero_neural_gradient_recovery_does_not_use_an_uncalibrated_partial_path():
    problem = _small_problem()
    surrogate = _UnusableGradientSurrogate()
    score = pathwise_case_score_details(
        physics=problem.physics, surrogate=surrogate, m=problem.m0,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
    )
    # There is no nonzero neural direction to calibrate. Recovery must rely
    # solely on a full exact gradient, regardless of a supplied positive scale.
    assert score.certificates_scored == 0 and score.score == 0
    result = adaptive_hybrid_direction(
        physics=problem.physics, surrogate=surrogate, m=problem.m0,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
        bound_scales={block.label: 1e-6 for block in problem.blocks},
    )
    assert result.certified_descent and result.certificate_rounds == 0
    assert result.exact_block_evaluations == len(problem.blocks)


def test_cancellation_after_repair_reuses_paid_exact_block(monkeypatch):
    from types import SimpleNamespace

    problem = _small_problem()
    first, second = problem.blocks
    original_gradient = problem.physics.objective_and_gradient
    _, first_gradient, _, _ = original_gradient(
        problem.m0, problem.physics.source(first.source_index), first.observed,
        problem.geometry.receiver_indices, first.frequency_hz,
    )
    exact_calls = []

    def counted_exact(m, q, *args):
        exact_calls.append(int(np.argmax(np.abs(q))))
        return original_gradient(m, q, *args)

    monkeypatch.setattr(problem.physics, "objective_and_gradient", counted_exact)

    class CancellingSurrogate(_UnusableGradientSurrogate):
        def objective_and_gradient(self, m, q, observed, receivers, frequency_hz):
            gradient = (
                2 * first_gradient if int(np.argmax(np.abs(q))) == first.source_index else
                -first_gradient
            )
            return 1.0, gradient, self.state(m, q, frequency_hz), -observed

    def forced_repair(**kwargs):
        first_source = int(np.argmax(np.abs(kwargs["q"]))) == first.source_index
        return SimpleNamespace(
            surrogate_directional_derivative=0.0,
            directional_error_bound=2.0 if first_source else 1.0,
        )

    monkeypatch.setattr("wavecert.repair.adaptive.certify_direction", forced_repair)
    result = adaptive_hybrid_direction(
        physics=problem.physics, surrogate=CancellingSurrogate(), m=problem.m0,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
    )
    assert result.mode == "zero-hybrid-gradient-exact-fallback"
    assert result.certified_descent
    assert exact_calls == [first.source_index, second.source_index]
    assert result.exact_block_evaluations == 2
    assert result.certificate_rounds == 1
    assert np.isfinite(result.exact_objective)


@pytest.mark.parametrize("method", [adaptive_hybrid_direction, global_fallback_direction])
def test_exact_fallback_never_certifies_a_direction_zeroed_by_tolerance(monkeypatch, method):
    problem = _small_problem()

    def tiny_gradient(m, q, *args):
        return 1.0, np.full_like(m, 1e-17), np.zeros_like(q), np.zeros(6)

    monkeypatch.setattr(problem.physics, "objective_and_gradient", tiny_gradient)
    result = method(
        physics=problem.physics, surrogate=_UnusableGradientSurrogate(), m=problem.m0,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
    )
    assert not result.certified_descent
    assert result.final_upper_derivative == 0
    np.testing.assert_array_equal(result.direction, np.zeros_like(problem.m0))
