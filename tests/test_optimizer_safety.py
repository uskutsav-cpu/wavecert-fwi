from dataclasses import replace

import numpy as np
import pytest

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.optimization.fwi import _backtrack, run_inversion
from wavecert.repair.adaptive import AdaptiveDirectionResult
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_projected_armijo_uses_actual_displacement_and_rejects_outward_direction():
    m = np.array([0.25, 0.5])
    calls = []

    def objective(candidate):
        calls.append(candidate.copy())
        return float(np.sum(candidate))

    candidate, step, value, accepted = _backtrack(
        m, np.array([-1.0, 0.1]), objective, float(np.sum(m)),
        initial_step=1.0, vmin=1.0, vmax=2.0, gradient=np.ones(2), max_trials=3,
    )
    # The unprojected slope is negative; the feasible displacement is ascent.
    assert not accepted
    np.testing.assert_array_equal(candidate, m)
    assert step == 0.0 and value == float(np.sum(m))
    assert calls == []


def test_armijo_requires_sufficient_decrease_and_failed_search_keeps_model():
    m = np.array([0.5])
    calls = []

    def objective(candidate):
        calls.append(candidate.copy())
        return 1.0 - 1e-12

    candidate, step, _, accepted = _backtrack(
        m, np.array([1.0]), objective, 1.0, initial_step=0.1,
        vmin=1.0, vmax=2.0, gradient=np.array([-1.0]), max_trials=2,
    )
    assert not accepted and step == 0.0
    np.testing.assert_array_equal(candidate, m)
    assert len(calls) == 2


def test_zero_projected_step_does_not_claim_acceptance():
    candidate, step, _, accepted = _backtrack(
        np.array([0.25]), np.array([-1.0]), lambda _: pytest.fail("zero step evaluated"),
        1.0, initial_step=0.1, vmin=1.0, vmax=2.0,
    )
    assert not accepted and step == 0.0
    np.testing.assert_array_equal(candidate, [0.25])


def _problem():
    return build_synthetic_problem(
        shape=(8, 8), frequencies_hz=(3.0,), n_sources=2, n_receivers=6
    )


@pytest.mark.parametrize("acceptance", ["exact-decrease", "exact-armijo"])
@pytest.mark.parametrize("policy", ["exact", "neural-only", "random-repair", "residual-repair", "periodic-exact"])
def test_verified_trajectories_never_increase_true_objective(policy, acceptance):
    problem = _problem()
    run = run_inversion(
        policy=policy, physics=problem.physics,
        surrogate=SmoothedHelmholtzSurrogate(problem.physics, smoothing_sigma=0.7),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=3,
        step_acceptance=acceptance, initial_step=0.1,
        repair_blocks=1, periodic_interval=2,
    )
    values = np.asarray([r.true_objective for r in run.records])
    assert np.all(np.diff(values) <= 1e-12)
    assert np.all((run.final_model >= 1 / 3.2**2) & (run.final_model <= 1 / 1.55**2))
    if acceptance == "exact-armijo" and policy not in {"exact", "periodic-exact"}:
        assert run.total_verification_gradient_blocks == 3 * len(problem.blocks)
    assert run.records[-1].mode == "final-evaluation"
    assert run.records[-1].line_search_trials == 0


def test_repair_baselines_are_seeded_and_replay_explicit_budget():
    problem = _problem()
    options = dict(
        policy="random-repair", m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=3,
        random_seed=79, repair_schedule=(0, 1, 2), step_acceptance="exact-decrease",
    )
    runs = []
    for _ in range(2):
        physics = replace(problem.physics)
        runs.append(run_inversion(
            physics=physics, surrogate=SmoothedHelmholtzSurrogate(physics), **options,
        ))
    np.testing.assert_array_equal(runs[0].final_model, runs[1].final_model)
    assert [r.exact_gradient_blocks for r in runs[0].records[:-1]] == [0, 1, 2]


def test_uncertified_adaptive_direction_is_never_used(monkeypatch):
    problem = _problem()
    result = AdaptiveDirectionResult(
        mode="budget-exhausted", certified_descent=False, direction=np.ones_like(problem.m0),
        repaired_blocks=(), exact_block_evaluations=0, total_blocks=len(problem.blocks),
        initial_upper_derivative=1.0, final_upper_derivative=1.0,
        surrogate_gradient_norm=1.0, final_hybrid_gradient_norm=1.0, certificate_rounds=1,
    )
    monkeypatch.setattr("wavecert.optimization.fwi.adaptive_hybrid_direction", lambda **_: result)
    run = run_inversion(
        policy="conformal-selective", physics=problem.physics,
        surrogate=SmoothedHelmholtzSurrogate(problem.physics),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        step_acceptance="exact-armijo",
    )
    np.testing.assert_array_equal(run.final_model, problem.m0)
    assert not run.records[0].accepted and not run.records[0].direction_certified
    assert run.records[0].mode == "rejected-budget-exhausted"
    assert run.total_verification_gradient_blocks == 0


def test_validation_costs_do_not_warm_online_solver_cache():
    problem = _problem()
    run = run_inversion(
        policy="exact", physics=problem.physics,
        surrogate=SmoothedHelmholtzSurrogate(problem.physics),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        step_acceptance="exact-armijo",
    )
    # Validation has independent factorizations; the online step must also pay.
    if not run.total_online_exact_cost:
        pytest.skip("physics accounting API unavailable")
    assert run.records[0].online_exact_cost["factorizations"] >= 1
    assert run.records[0].validation_exact_cost["factorizations"] >= 1
    assert run.total_online_exact_cost["adjoint_rhs"] == len(problem.blocks)
    assert run.total_validation_exact_cost["adjoint_rhs"] == 0
    assert run.total_online_exact_cost["forward_rhs"] >= 2 * len(problem.blocks)
    assert run.total_validation_exact_cost["forward_rhs"] == 2 * len(problem.blocks)


@pytest.mark.parametrize("kwargs", [
    {"step_acceptance": "magic"}, {"iterations": -1}, {"repair_blocks": -1},
    {"periodic_interval": 0}, {"repair_schedule": (1,)}, {"vmin": 3, "vmax": 2},
])
def test_invalid_optimizer_configuration_fails_early(kwargs):
    problem = _problem()
    options = dict(
        policy="random-repair", physics=problem.physics,
        surrogate=SmoothedHelmholtzSurrogate(problem.physics),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=2,
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        run_inversion(**options)


def test_wrong_neural_gradient_cannot_pass_true_armijo():
    problem = _problem()

    class ReversedGradient:
        def state(self, m, q, frequency_hz):
            return problem.physics.solve_state(m, q, frequency_hz)

        def objective_and_gradient(self, m, q, observed, receivers, frequency_hz):
            value, gradient, state, residual = problem.physics.objective_and_gradient(
                m, q, observed, receivers, frequency_hz,
            )
            return value, -gradient, state, residual

    run = run_inversion(
        policy="neural-only", physics=problem.physics, surrogate=ReversedGradient(),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        step_acceptance="exact-armijo", initial_step=0.01,
    )
    np.testing.assert_array_equal(run.final_model, problem.m0)
    assert not run.records[0].accepted
    assert run.records[0].verification_gradient_blocks == len(problem.blocks)
    assert run.records[0].line_search_trials == 0


def test_fully_repaired_adaptive_gradient_is_reused_for_armijo(monkeypatch):
    problem = _problem()
    from wavecert.optimization.fwi import exact_objective_gradient

    _, gradient = exact_objective_gradient(
        problem.physics, problem.m0, problem.blocks, problem.geometry.receiver_indices,
    )
    norm = float(np.linalg.norm(gradient))
    result = AdaptiveDirectionResult(
        mode="exact-fallback", certified_descent=True, direction=-gradient / norm,
        repaired_blocks=tuple(b.label for b in problem.blocks),
        exact_block_evaluations=len(problem.blocks), total_blocks=len(problem.blocks),
        initial_upper_derivative=1.0, final_upper_derivative=-norm,
        surrogate_gradient_norm=1.0, final_hybrid_gradient_norm=norm, certificate_rounds=2,
    )
    monkeypatch.setattr("wavecert.optimization.fwi.adaptive_hybrid_direction", lambda **_: result)
    run = run_inversion(
        policy="deterministic-selective", physics=problem.physics,
        surrogate=SmoothedHelmholtzSurrogate(problem.physics),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        step_acceptance="exact-armijo",
    )
    assert run.total_verification_gradient_blocks == 0
    assert run.records[0].direction_certified
    assert run.records[0].guarantee_scope == "discrete-deterministic-direction"


def test_legacy_surrogate_decrease_can_increase_true_objective_but_verification_blocks_it():
    """A smooth, self-consistent surrogate merit can favor an ascent direction."""
    from wavecert.optimization.fwi import exact_objective_gradient

    problem = _problem()
    _, gradient = exact_objective_gradient(
        problem.physics, problem.m0, problem.blocks, problem.geometry.receiver_indices,
    )
    ascent = gradient / np.linalg.norm(gradient)
    reference_model = problem.m0.copy()
    receivers = problem.geometry.receiver_indices
    observations = {
        (b.source_index, b.frequency_hz): np.asarray(b.observed) for b in problem.blocks
    }

    class AffineSurrogate:
        def _scalar(self, m):
            return 1.0 - float(np.dot(ascent, m - reference_model))

        def state(self, m, q, frequency_hz):
            scalar = self._scalar(m)
            source = int(np.argmax(np.abs(q)))
            state = np.zeros_like(q, dtype=complex)
            state[list(receivers)] = observations[(source, frequency_hz)] + scalar / np.sqrt(len(receivers))
            return state

        def objective_and_gradient(self, m, q, observed, receiver_indices, frequency_hz):
            scalar = self._scalar(m)
            state = self.state(m, q, frequency_hz)
            residual = state[list(receiver_indices)] - observed
            return 0.5 * scalar**2, -scalar * ascent, state, residual

    options = dict(
        policy="neural-only", surrogate=AffineSurrogate(), m0=problem.m0,
        m_true=problem.m_true, blocks=problem.blocks, receiver_indices=receivers,
        iterations=1, initial_step=0.001, max_line_search_trials=4,
    )
    legacy = run_inversion(physics=replace(problem.physics), step_acceptance="surrogate", **options)
    verified = run_inversion(physics=replace(problem.physics), step_acceptance="exact-decrease", **options)
    assert legacy.records[0].accepted
    assert legacy.records[-1].true_objective > legacy.records[0].true_objective
    assert not verified.records[0].accepted
    np.testing.assert_array_equal(verified.final_model, problem.m0)
    assert verified.records[-1].true_objective == verified.records[0].true_objective


@pytest.mark.parametrize("mismatch", ["frequency", "source", "receivers"])
@pytest.mark.parametrize("acceptance", ["surrogate", "exact-decrease"])
def test_acquisition_guard_abstains_without_any_surrogate_call(mismatch, acceptance):
    problem = _problem()
    domain = {
        "frequencies_hz": [3.0],
        "source_indices": [int(b.source_index) for b in problem.blocks],
        "receiver_indices": list(problem.geometry.receiver_indices),
    }
    if mismatch == "frequency":
        domain["frequencies_hz"] = [2.5]
    elif mismatch == "source":
        domain["source_indices"] = [int(problem.blocks[0].source_index)]
    else:
        domain["receiver_indices"] = list(reversed(problem.geometry.receiver_indices))

    class NoSurrogateCalls:
        def __getattr__(self, name):
            pytest.fail(f"acquisition mismatch called surrogate method {name}")

    run = run_inversion(
        policy="conformal-selective-guarded", physics=problem.physics,
        surrogate=NoSurrogateCalls(), m0=problem.m0, m_true=problem.m_true,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
        iterations=2, step_acceptance=acceptance, calibration_domain=domain,
    )
    assert all(r.mode == "domain-mismatch-exact-fallback" for r in run.records[:-1])
    assert all(r.exact_gradient_blocks == len(problem.blocks) for r in run.records[:-1])
    assert run.total_online_exact_cost["stability_rhs"] == 0
    assert run.total_online_exact_cost["adjoint_rhs"] == 2 * len(problem.blocks)
    assert run.records[-1].true_objective < run.records[0].true_objective
    assert all(r.guarantee_scope == "exact-gradient-direction" for r in run.records[:-1])


def test_acquisition_guard_preserves_within_support_calibration_flow(monkeypatch):
    problem = _problem()
    calls = []
    result = AdaptiveDirectionResult(
        mode="budget-exhausted", certified_descent=False, direction=np.ones_like(problem.m0),
        repaired_blocks=(), exact_block_evaluations=0, total_blocks=len(problem.blocks),
        initial_upper_derivative=1.0, final_upper_derivative=1.0,
        surrogate_gradient_norm=1.0, final_hybrid_gradient_norm=1.0, certificate_rounds=1,
    )

    def adaptive(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr("wavecert.optimization.fwi.adaptive_hybrid_direction", adaptive)
    scales = {b.label: 0.1 for b in problem.blocks}
    run = run_inversion(
        policy="conformal-selective-guarded", physics=problem.physics,
        surrogate=SmoothedHelmholtzSurrogate(problem.physics),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        block_scales=scales, calibration_domain={
            "frequencies_hz": [3.0],
            "source_indices": [int(b.source_index) for b in problem.blocks],
            "receiver_indices": list(problem.geometry.receiver_indices),
        },
    )
    assert len(calls) == 1
    assert calls[0]["bound_scales"] == scales
    assert run.records[0].guarantee_scope == "calibrated-direction"
    assert not run.records[0].accepted


@pytest.mark.parametrize("domain", [None, {}, {"frequencies_hz": [3.0]}, {
    "frequencies_hz": [], "source_indices": [0], "receiver_indices": [0],
}, {
    "frequencies_hz": [3.0], "source_indices": [True], "receiver_indices": [0],
}])
def test_acquisition_guard_requires_a_valid_nonempty_contract(domain):
    problem = _problem()
    with pytest.raises(ValueError, match="calibration"):
        run_inversion(
            policy="conformal-selective-guarded", physics=problem.physics,
            surrogate=SmoothedHelmholtzSurrogate(problem.physics),
            m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
            receiver_indices=problem.geometry.receiver_indices, iterations=1,
            calibration_domain=domain,
        )


def test_unavailable_analytic_preflight_falls_back_before_any_surrogate_work(monkeypatch):
    from types import SimpleNamespace

    problem = _problem()
    monkeypatch.setattr(
        problem.physics, "analytical_stability_bound",
        lambda *args: SimpleNamespace(beta_lower_bound=0.0),
    )

    class NoSurrogateCalls:
        def __getattr__(self, name):
            pytest.fail(f"unavailable analytical bound called surrogate method {name}")

    run = run_inversion(
        policy="analytic-selective", physics=problem.physics, surrogate=NoSurrogateCalls(),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        step_acceptance="surrogate",
    )
    assert run.records[0].mode == "unavailable-stability-exact-fallback"
    assert run.records[0].exact_gradient_blocks == len(problem.blocks)
    assert run.records[0].online_exact_cost["stability_rhs"] == 0
    assert run.records[0].online_exact_cost["svd_evaluations"] == 0
    assert run.records[-1].true_objective < run.records[0].true_objective


@pytest.mark.parametrize("policy", ["analytic-selective", "deterministic-global"])
def test_full_fallback_reuses_paid_current_objective_without_extra_forward(monkeypatch, policy):
    from types import SimpleNamespace

    problem = _problem()
    monkeypatch.setattr(
        problem.physics, "analytical_stability_bound",
        lambda *args: SimpleNamespace(beta_lower_bound=0.0),
    )
    exact_calls = []
    exact_gradient = problem.physics.objective_and_gradient

    def counted_gradient(*args, **kwargs):
        exact_calls.append(1)
        return exact_gradient(*args, **kwargs)

    monkeypatch.setattr(problem.physics, "objective_and_gradient", counted_gradient)

    class ConstantSurrogate:
        def state(self, m, q, frequency_hz):
            return np.zeros_like(q, dtype=complex)

        def jvp(self, m, direction, q, frequency_hz):
            return np.zeros_like(q, dtype=complex)

        def objective_and_gradient(self, m, q, observed, receivers, frequency_hz):
            return 0.5 * float(np.vdot(observed, observed).real), np.ones_like(m), self.state(m, q, frequency_hz), -observed

    run = run_inversion(
        policy=policy, physics=problem.physics, surrogate=ConstantSurrogate(),
        m0=problem.m0, m_true=problem.m_true, blocks=problem.blocks,
        receiver_indices=problem.geometry.receiver_indices, iterations=1,
        step_acceptance="exact-decrease",
    )
    record = run.records[0]
    assert record.exact_gradient_blocks == len(problem.blocks)
    assert len(exact_calls) == len(problem.blocks)
    assert record.accepted
    # One current forward per already-paid exact gradient block, then only
    # candidate forward solves for the recorded line-search trials.
    assert record.online_exact_cost["forward_rhs"] == len(problem.blocks) * (1 + record.line_search_trials)
