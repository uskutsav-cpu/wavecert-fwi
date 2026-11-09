import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wavecert.optimization import paid_descent as engine

DirectionProposal = engine.DirectionProposal
TrajectoryOptions = engine.TrajectoryOptions


class ManualClock:
    """Abstract work clock only: these tests collect no benchmark timings."""

    def __init__(self):
        self.time = 0.

    def __call__(self):
        return self.time

    def advance(self, amount):
        self.time += amount


class ToyOracle:
    def __init__(self, clock, target, *, affine=False, setup=1., fail_on=None):
        self.clock, self.target, self.affine = clock, np.asarray(target), affine
        self.fail_on, self.cache, self.current = fail_on, {}, None
        self.counts = dict(constructions=1, merits=0, merit_hits=0, gradients=0, gradient_hits=0, accepts=0)
        clock.advance(setup)

    def evaluate(self, model):
        model = np.asarray(model)
        key = model.tobytes()
        if key in self.cache:
            self.counts["merit_hits"] += 1
            return self.cache[key]
        self.counts["merits"] += 1
        self.clock.advance(1.)
        if self.fail_on == self.counts["merits"]:
            raise RuntimeError("injected accurate-linear-solve failure")
        objective = float(np.sum(model)) if self.affine else .5*float(np.sum((model-self.target)**2))
        receipt = SimpleNamespace(model=model.copy(), objective=objective, owner=self, g=None)
        self.cache[key] = receipt
        return receipt

    def gradient(self, receipt):
        assert receipt.owner is self
        if receipt.g is None:
            self.counts["gradients"] += 1
            self.clock.advance(2.)
            receipt.g = np.ones_like(receipt.model) if self.affine else receipt.model-self.target
        else:
            self.counts["gradient_hits"] += 1
        return SimpleNamespace(gradient=receipt.g, receipt=receipt)

    def accept(self, receipt):
        assert receipt.owner is self
        self.counts["accepts"] += 1
        self.current = receipt

    def cost_snapshot(self):
        return dict(self.counts)


class ToyProxy:
    def __init__(self, clock, direction, **kwargs):
        self.clock, self.direction, self.kwargs = clock, direction, kwargs
        self.calls = 0

    def propose(self, model, paid_merit):
        self.calls += 1
        self.clock.advance(.1)
        assert np.array_equal(model, paid_merit.model)  # Only its paid current receipt.
        return DirectionProposal(np.asarray(self.direction), **self.kwargs)

    def cost_snapshot(self):
        return dict(proposals=self.calls)


def _options(**changes):
    base = TrajectoryOptions(budget_seconds=100., maximum_steps=5, initial_step=.1, maximum_step=.1,
                             accepted_step_growth=1., maximum_line_search_trials=3)
    return replace(base, **changes)


def test_exact_and_raw_good_proxy_use_common_paid_merit_and_reach_same_models_with_complete_costs():
    results = []
    for raw in (False, True):
        clock = ManualClock()
        def oracle(clock=clock):
            return ToyOracle(clock, [.3])

        proxy = (lambda _paid_oracle, clock=clock: ToyProxy(clock, [-1.])) if raw else None
        result = engine.run_paid_descent(oracle_factory=oracle, proxy_factory=proxy, m0=np.array([.8]),
            lower_bound=.1, upper_bound=1., options=_options(relative_objective_target=.01), clock=clock)
        results.append(result)
        assert result["status"] == "objective_target"
        assert result["costs_complete"] and not result["truth_used_for_policy"]
        values = [row["objective"] for row in result["checkpoints"]]
        assert np.all(np.diff(values) < 0)
        assert all(step["accepted"] and step["trials"][-1]["accepted"] for step in result["steps"])
        assert result["total_online_costs"]["oracle"]["merits"] == 6
        assert result["total_online_costs"]["oracle"]["gradients"] == (0 if raw else 5)
        json.dumps(engine.trajectory_to_dict(result), allow_nan=False)
    assert np.array_equal(results[0]["final_model"], results[1]["final_model"])


def test_raw_ascent_pays_rejected_trials_then_one_exact_recovery_and_counts_every_call():
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3]),
        proxy_factory=lambda _paid_oracle: ToyProxy(clock, [1.]), m0=np.array([.8]), lower_bound=.1, upper_bound=1.,
        options=_options(maximum_steps=1), clock=clock)
    assert result["status"] == "maximum_steps"
    step = result["steps"][0]
    assert [trial["accepted"] for trial in step["trials"]] == [False, False, False, True]
    assert step["exact_recoveries"][0]["reason"] == "proxy_line_search_exhausted"
    assert result["total_online_costs"]["oracle"]["merits"] == 5
    assert result["total_online_costs"]["oracle"]["gradients"] == 1
    assert result["total_online_costs"]["proposal"]["proposals"] == 1
    assert result["final_objective"] < result["initial_objective"]


def test_raw_direction_certificate_cannot_authorize_ascent_after_projection():
    clock = ManualClock()
    raw = np.array([-1., .1])
    assert np.ones(2) @ raw < 0
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [0., 0.], affine=True),
        proxy_factory=lambda _paid_oracle: ToyProxy(clock, raw, certified_descent=True,
            certificate_direction=raw.copy(), certificate_upper=-.8),
        m0=np.array([.25, .5]), lower_bound=.25, upper_bound=1.,
        options=_options(maximum_steps=1, initial_step=.2, maximum_step=.2, maximum_line_search_trials=2), clock=clock)
    trials = result["steps"][0]["trials"]
    assert trials[0]["projected_direction_changed"]
    assert trials[0]["raw_direction_certificate_does_not_certify_projected_displacement"]
    assert trials[0]["objective"] > result["initial_objective"] and not trials[0]["accepted"]
    assert trials[-1]["exact_direction"] and trials[-1]["accepted"]
    assert result["final_objective"] < result["initial_objective"]


@pytest.mark.parametrize("direction", [[0.], [np.nan], [np.inf]])
def test_zero_or_nonfinite_proxy_is_not_stationarity_and_uses_paid_exact_recovery(direction):
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3]),
        proxy_factory=lambda _paid_oracle: ToyProxy(clock, direction), m0=np.array([.8]), lower_bound=.1, upper_bound=1.,
        options=_options(maximum_steps=1), clock=clock)
    assert result["steps"][0]["accepted"]
    assert result["steps"][0]["exact_recoveries"][0]["reason"] == "zero_or_invalid_proxy_direction"
    assert result["total_online_costs"]["oracle"]["gradients"] == 1
    json.dumps(engine.trajectory_to_dict(result), allow_nan=False)


def test_certificate_abstention_and_invalid_certificate_both_pay_exact_fallback():
    for kwargs, reason in [
        (dict(certified_descent=False), "certificate_abstention"),
        (dict(certified_descent=True, certificate_direction=np.array([1.]), certificate_upper=-1.),
         "invalid_raw_direction_certificate"),
    ]:
        clock = ManualClock()
        result = engine.run_paid_descent(oracle_factory=lambda clock=clock: ToyOracle(clock, [.3]),
            proxy_factory=lambda _paid_oracle, kwargs=kwargs, clock=clock: ToyProxy(clock, [-1.], **kwargs), m0=np.array([.8]),
            lower_bound=.1, upper_bound=1., options=_options(maximum_steps=1), clock=clock)
        assert result["steps"][0]["exact_recoveries"][0]["reason"] == reason
        assert result["total_online_costs"]["oracle"]["gradients"] == 1


@pytest.mark.parametrize("flags", [dict(certified_descent=np.bool_(False)),
    dict(fallback_required=np.bool_(True))])
def test_numpy_boolean_abstention_cannot_bypass_paid_exact_fallback(flags):
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3]),
        proxy_factory=lambda _paid_oracle: ToyProxy(clock, [-1.], **flags),
        m0=np.array([.8]), lower_bound=.1, upper_bound=1., options=_options(maximum_steps=1), clock=clock)
    assert result["steps"][0]["exact_recoveries"][0]["reason"] == "certificate_abstention"
    assert result["total_online_costs"]["oracle"]["gradients"] == 1


@pytest.mark.parametrize("flags", [dict(certified_descent="false"), dict(fallback_required="false")])
def test_non_boolean_certificate_flags_fail_closed_without_a_trial(flags):
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3]),
        proxy_factory=lambda _paid_oracle: ToyProxy(clock, [-1.], **flags),
        m0=np.array([.8]), lower_bound=.1, upper_bound=1., options=_options(maximum_steps=1), clock=clock)
    assert result["status"] == "failed" and result["steps"][0]["trials"] == []
    assert result["total_online_costs"]["oracle"]["merits"] == 1
    assert "TypeError" in result["message"]


def test_equal_native_lbfgsb_callback_is_paid_but_receives_no_strict_endpoint_credit(monkeypatch):
    clock = ManualClock()
    candidate = np.array([.3, .8])  # Same quadratic objective as [.8, .3].

    def equal_callback_solver(function, initial, *, jac, method, bounds, callback, options):
        del initial, method, bounds, options
        function(candidate)
        jac(candidate)
        callback(candidate)
        return SimpleNamespace(success=True, status=0, message="native equal callback fixture",
            nit=1, nfev=1, njev=1)

    monkeypatch.setattr(engine, "minimize", equal_callback_solver)
    initial = np.array([.8, .3])
    result = engine.run_exact_lbfgsb(oracle_factory=lambda: ToyOracle(clock, [0., 0.]),
        m0=initial, lower_bound=.1, upper_bound=1., options=_options(), clock=clock)
    assert result["status"] == "solver_success" and result["solver_result"]["nit"] == 1
    assert np.array_equal(result["final_model"], initial) and len(result["checkpoints"]) == 1
    assert not result["steps"][0]["accepted"]
    assert not result["steps"][0]["native_callback"]["strict_decrease_endpoint_eligible"]
    assert result["total_online_costs"]["oracle"]["merits"] == 2
    assert result["total_online_costs"]["oracle"]["gradients"] == 1
    assert "equal callbacks remain paid" in result["acceptance"]


def test_equal_native_callback_finishing_after_budget_stops_before_solver_success(monkeypatch):
    clock = ManualClock()

    def callback_only_solver(function, initial, *, jac, method, bounds, callback, options):
        del function, initial, jac, method, bounds, options
        # No preceding fun/jac call: callback's own accurate merit must still
        # enforce the post-work cap, including the equal-objective branch.
        callback(np.array([.3, .8]))
        pytest.fail("an over-budget native callback must interrupt the solver")

    monkeypatch.setattr(engine, "minimize", callback_only_solver)
    result = engine.run_exact_lbfgsb(oracle_factory=lambda: ToyOracle(clock, [0., 0.]),
        m0=np.array([.8, .3]), lower_bound=.1, upper_bound=1.,
        options=_options(budget_seconds=2.5), clock=clock)
    assert result["status"] == "budget_exhausted" and result["solver_result"] is None
    assert result["total_online_costs"]["oracle"]["merits"] == 2
    assert len(result["checkpoints"]) == 1 and result["budget_overshoot_seconds"] == .5
    assert not result["steps"][0]["native_callback"]["strict_decrease_endpoint_eligible"]


def test_only_paid_exact_projected_gradient_can_establish_constrained_stationarity():
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [0.]),
        proxy_factory=lambda _paid_oracle: ToyProxy(clock, [0.]), m0=np.array([.25]), lower_bound=.25, upper_bound=1.,
        options=_options(), clock=clock)
    assert result["status"] == "exact_projected_stationarity"
    assert result["total_online_costs"]["oracle"]["gradients"] == 1
    assert result["steps"][0]["paid_exact_projected_gradient_mapping_inf"] == 0
    assert result["total_online_costs"]["oracle"]["merits"] == 1


def test_budget_overrun_retains_paid_work_without_post_budget_model_credit():
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3]), proxy_factory=None,
        m0=np.array([.8]), lower_bound=.1, upper_bound=1., options=_options(budget_seconds=4.5), clock=clock)
    assert result["status"] == "budget_exhausted"
    assert result["total_online_seconds"] == 5 and result["budget_overshoot_seconds"] == .5
    assert np.array_equal(result["final_model"], [.8])
    assert result["steps"][0]["trials"][-1]["objective"] < result["initial_objective"]
    assert not result["steps"][0]["trials"][-1]["accepted"]
    assert result["steps"][0]["trials"][-1]["reason"] == "merit_completed_after_budget"
    assert result["total_online_costs"]["oracle"]["merits"] == 2
    assert all(row["elapsed_seconds"] <= 4.5 for row in result["checkpoints"] if row["kind"] == "accepted")


def test_budget_during_setup_cannot_borrow_an_offline_initial_objective():
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3], setup=2.), proxy_factory=None,
        m0=np.array([.8]), lower_bound=.1, upper_bound=1., options=_options(budget_seconds=1.), clock=clock)
    assert result["status"] == "budget_exhausted" and result["final_objective"] is None
    assert not result["checkpoints"] and result["total_online_costs"]["oracle"]["merits"] == 0


def test_failed_linear_solve_is_retained_as_failure_with_paid_attempt_and_last_model():
    clock = ManualClock()
    result = engine.run_paid_descent(oracle_factory=lambda: ToyOracle(clock, [.3], fail_on=2), proxy_factory=None,
        m0=np.array([.8]), lower_bound=.1, upper_bound=1., options=_options(), clock=clock)
    assert result["status"] == "failed" and "linear-solve failure" in result["message"]
    assert result["total_online_costs"]["oracle"]["merits"] == 2
    assert np.array_equal(result["final_model"], [.8])
    json.dumps(engine.trajectory_to_dict(result), allow_nan=False)


def test_exact_lbfgsb_uses_same_merit_receipts_and_preserves_native_termination():
    clock = ManualClock()
    result = engine.run_exact_lbfgsb(oracle_factory=lambda: ToyOracle(clock, [.3]), m0=np.array([.8]),
        lower_bound=.1, upper_bound=1., options=_options(maximum_steps=20), clock=clock)
    assert result["status"] == "solver_success" and result["solver_result"]["success"]
    assert np.allclose(result["final_model"], [.3], atol=1e-10)
    assert result["total_online_costs"]["oracle"]["merit_hits"] > 0
    assert result["total_online_costs"]["oracle"]["merits"] == result["total_online_costs"]["oracle"]["gradients"]
    assert any(step.get("library_evaluations") for step in result["steps"])
    assert sum(step["online_cost_delta"]["oracle"]["gradients"] for step in result["steps"]) == result["total_online_costs"]["oracle"]["gradients"]
    json.dumps(engine.trajectory_to_dict(result), allow_nan=False)


def test_exact_lbfgsb_budget_does_not_credit_an_unfinished_trial():
    clock = ManualClock()
    result = engine.run_exact_lbfgsb(oracle_factory=lambda: ToyOracle(clock, [.3]), m0=np.array([.8]),
        lower_bound=.1, upper_bound=1., options=_options(budget_seconds=4.5, maximum_steps=20), clock=clock)
    assert result["status"] == "budget_exhausted"
    assert np.array_equal(result["final_model"], [.8])
    assert result["total_online_seconds"] > result["budget_seconds"]
    assert result["total_online_costs"]["oracle"]["merits"] == 2


@pytest.mark.parametrize("change", [dict(initial_step=0.), dict(relative_objective_target=np.nan),
                                   dict(maximum_line_search_trials=0), dict(budget_seconds=-1.)])
def test_invalid_design_options_fail_before_any_factory_work(change):
    clock = ManualClock()
    with pytest.raises(ValueError):
        engine.run_paid_descent(oracle_factory=lambda: pytest.fail("invalid design constructed oracle"),
            proxy_factory=None, m0=np.array([.8]), lower_bound=.1, upper_bound=1.,
            options=_options(**change), clock=clock)
