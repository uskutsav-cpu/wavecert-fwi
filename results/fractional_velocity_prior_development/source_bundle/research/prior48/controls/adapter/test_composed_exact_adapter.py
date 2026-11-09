"""Manufactured scalar objectives only: no project arrays or wave/PDE calls."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import composed_exact_adapter as adapter


class Clock:
    """Synthetic work units, never measured performance evidence."""
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, value):
        self.value += value


@dataclass(eq=False)
class ToyPoint:
    model: np.ndarray
    objective: float
    owner: object
    gradient_cache: object = None

    def to_dict(self):
        return {"kind": "manufactured_scalar_fixture_only", "objective": self.objective,
                    "model_sha256": adapter.model_sha(self.model)}


class ToyNative:
    """Explicit manufactured delegate; this is not a native PDE receipt issuer."""
    def __init__(self, clock, *, target=.1, n=1, setup=.1, merit=.2, gradient=.3,
                 late_second_accept=0.0, objective_scale=1.0, interrupt_gradient=False):
        self.clock, self.target, self.n = clock, target, n
        self.merit_time, self.gradient_time = merit, gradient
        self.late_second_accept = late_second_accept
        self.scale, self.interrupt_gradient = objective_scale, interrupt_gradient
        self.current = self.trial = None
        self.issued, self.events, self.accepted = set(), [], []
        self.counts = {"constructions": 1, "evaluate_calls": 0, "objective_evaluations": 0,
            "objective_cache_hits": 0, "gradient_calls": 0, "gradient_evaluations": 0,
            "gradient_cache_hits": 0, "accept_calls": 0, "cost_snapshot_calls": 0}
        self.clock.advance(setup)

    def evaluate(self, model):
        model = np.asarray(model)
        self.events.append(("evaluate", adapter.model_sha(model)))
        self.counts["evaluate_calls"] += 1
        for point in (self.current, self.trial):
            if point is not None and np.array_equal(point.model, model):
                self.counts["objective_cache_hits"] += 1
                return point
        self.clock.advance(self.merit_time)
        self.counts["objective_evaluations"] += 1
        point = ToyPoint(adapter.immutable(model), .5*float(np.sum((model-self.target)**2))/self.scale, self)
        self.issued.add(point)
        self.trial = point
        return point

    def gradient(self, point):
        assert point in self.issued and point.owner is self
        self.events.append(("gradient", adapter.model_sha(point.model)))
        self.counts["gradient_calls"] += 1
        if point.gradient_cache is None:
            self.clock.advance(self.gradient_time)
            self.counts["gradient_evaluations"] += 1
            if self.interrupt_gradient:
                raise KeyboardInterrupt("manufactured in-flight gradient interruption")
            point.gradient_cache = adapter.immutable((point.model-self.target)/self.scale)
        else:
            self.counts["gradient_cache_hits"] += 1
        return SimpleNamespace(gradient=point.gradient_cache, receipt=point)

    def accept(self, point):
        assert point in self.issued and point.owner is self
        self.events.append(("accept", adapter.model_sha(point.model)))
        self.counts["accept_calls"] += 1
        if self.counts["accept_calls"] == 2:
            self.clock.advance(self.late_second_accept)
        # Like the inspected native producer, authorization only; no raw-Phi gate.
        self.current, self.trial = point, None
        self.accepted.append(point.model.copy())

    def cost_snapshot(self):
        self.events.append(("snapshot",))
        self.counts["cost_snapshot_calls"] += 1
        return {"manufactured_native": dict(self.counts)}


def scripted(actions):
    def solve(function, initial, *, jac, method, bounds, callback, options):
        assert method == "L-BFGS-B" and len(bounds) == initial.size
        values, callbacks = [], 0
        for kind, value in actions:
            model = np.full_like(initial, value)
            values.append(function(model))
            if kind != "merit_only":
                jac(model)
            if kind == "accept":
                callback(model)
                callbacks += 1
        solve.values = values
        return SimpleNamespace(success=True, status=0, message="manufactured callable sequence",
                               nit=callbacks, nfev=len(values), njev=len(values))
    solve.values = []
    return solve


def run(*, weight=100, actions=(), clock=None, budget=100, maximum_steps=10,
        dependencies=None, native_options=None):
    dependencies = dependencies or adapter.load_frozen_dependencies()
    if actions is not None:
        dependencies.engine.minimize = scripted(actions)
    clock = clock or Clock()
    made = []

    def factory():
        owner = ToyNative(clock, **(native_options or {}))
        made.append(owner)
        return owner

    options = dependencies.engine.TrajectoryOptions(budget_seconds=budget,
        maximum_steps=maximum_steps, initial_step=.002, maximum_step=.02)
    result = adapter.run_composed_exact(native_oracle_factory=factory,
        initial_model=np.array([.25]), shape=(1,1), grid_spacing=.025,
        smoothing_length=.1, weight=weight, dependencies=dependencies, options=options,
        global_started=0., clock=clock)
    return result, made[0], dependencies, clock


@pytest.mark.parametrize("weight", [0., 100.])
def test_constructor_and_initial_normalization_use_one_paid_merit_no_gradient(weight):
    result, native, _deps, _clock = run(weight=weight, maximum_steps=0)
    assert result["status"] == "maximum_steps"
    assert native.counts["evaluate_calls"] == native.counts["objective_evaluations"] == 1
    assert native.counts["gradient_calls"] == 0
    data0 = .5*(.25-.1)**2
    assert result["adapter"]["initial_paid_Phi0"] == data0
    assert result["initial_objective"] == 1.
    counts = result["adapter"]["adapter_counts"]
    assert counts["prior_constructions_completed"] == 1
    assert counts["prior_evaluations_completed"] == int(weight > 0)
    assert counts["normalization_initializations"] == 1
    assert result["adapter"]["initial_native_m_sha256"] == adapter.model_sha(np.array([.25]))


def test_lambda0_native_query_sequence_matches_analytic_normalized_reference():
    actions = [("trial",.15),("accept",.2)]
    result, native, _deps, _clock = run(weight=0, actions=actions)
    dependencies = adapter.load_frozen_dependencies()
    dependencies.engine.minimize = scripted(actions)
    clock = Clock()
    reference = ToyNative(clock, setup=.1, objective_scale=.5*(.25-.1)**2)
    options = dependencies.engine.TrajectoryOptions(budget_seconds=100., maximum_steps=10,
        initial_step=.002, maximum_step=.02)
    reference_result = dependencies.engine.run_exact_lbfgsb(oracle_factory=lambda:reference,
        m0=np.array([.25]), lower_bound=1/36, upper_bound=1., options=options,
        clock=adapter.GlobalAnchorClock(clock, 0.))
    assert native.events == reference.events
    assert native.counts == reference.counts
    assert result["final_objective"] == reference_result["final_objective"]
    assert np.array_equal(result["final_model"], reference_result["final_model"])
    assert result["adapter"]["adapter_counts"]["prior_evaluations_completed"] == 0


def test_regularized_acceptance_can_increase_raw_Phi_while_decreasing_composed_merit():
    result, native, dependencies, _clock = run(actions=[("accept",.21),("accept",.23)])
    assert result["status"] == "solver_success"
    assert len(native.accepted) == 3 and np.array_equal(result["final_model"], [.23])
    checkpoints = result["checkpoints"]
    assert checkpoints[2]["objective"] < checkpoints[1]["objective"] < checkpoints[0]["objective"]
    assert .5*(.23-.1)**2 > .5*(.21-.1)**2
    header = result["adapter"]["current_accepted"]
    assert header["native_data_receipt_header"]["objective"] == .5*(.23-.1)**2
    assert header["objective"] == checkpoints[-1]["objective"]
    assert result["certificate_scope"] is None and result["certificate_claim"] is False
    assert dependencies.engine.minimize.values[1] < dependencies.engine.minimize.values[0]
    baseline, base_native, _deps, _clock = run(weight=0, actions=[("accept",.21),("accept",.23)])
    assert baseline["status"] == "failed" and np.array_equal(baseline["final_model"], [.21])
    assert len(base_native.accepted) == 2


def test_rejected_trial_native_work_and_prior_arithmetic_are_separate_and_paid():
    result, native, _deps, _clock = run(actions=[("trial",.15),("accept",.23)])
    counts = result["adapter"]["adapter_counts"]
    assert native.counts["evaluate_calls"] == counts["native_evaluate_dispatches"] == 6
    assert native.counts["objective_evaluations"] == 3
    assert native.counts["gradient_calls"] == counts["native_gradient_dispatches"] == 2
    assert counts["prior_evaluations_completed"] == counts["objective_scalings"] == 3
    assert counts["gradient_compositions"] == 2
    assert counts["composed_point_cache_hits"] == 3
    assert not any(np.array_equal(m,[.15]) for m in native.accepted)


def test_paid_prior_constructor_and_preparation_can_exhaust_global_deadline_before_merit():
    dependencies = adapter.load_frozen_dependencies()
    clock = Clock(2.4)
    original = dependencies.prior.FractionalVelocityPrior.__init__

    def paid_init(self, *args, **kwargs):
        clock.advance(.8)
        original(self, *args, **kwargs)

    dependencies.prior.FractionalVelocityPrior.__init__ = paid_init
    result, native, _deps, _clock = run(dependencies=dependencies, clock=clock, budget=3.)
    assert result["status"] == "budget_exhausted" and not result["checkpoints"]
    assert native.counts["evaluate_calls"] == native.counts["gradient_calls"] == 0
    assert result["setup_seconds"] == pytest.approx(3.3)
    assert result["adapter"]["adapter_counts"]["prior_constructions_completed"] == 1


def test_late_trial_prior_keeps_initial_endpoint_and_counts_all_completed_work():
    dependencies = adapter.load_frozen_dependencies()
    clock = Clock(2.)
    original = dependencies.prior.FractionalVelocityPrior.evaluate
    calls = []

    def paid_prior(self, model):
        calls.append(1)
        clock.advance(.1 if len(calls)==1 else 1.0)
        return original(self, model)

    dependencies.prior.FractionalVelocityPrior.evaluate = paid_prior
    result, native, _deps, _clock = run(actions=[("accept",.23)], dependencies=dependencies,
                                      clock=clock, budget=3.)
    assert result["status"] == "budget_exhausted"
    assert np.array_equal(result["final_model"], [.25]) and len(native.accepted) == 1
    assert native.counts["objective_evaluations"] == 2 and native.counts["gradient_calls"] == 0
    assert result["adapter"]["adapter_counts"]["prior_evaluations_completed"] == 2
    assert result["total_online_seconds"] > 3.
    assert all(c["model_sha256"] == adapter.model_sha(np.array([.25])) for c in result["checkpoints"])


def test_late_native_promotion_restores_prior_eligible_owned_point():
    result, native, _deps, _clock = run(actions=[("accept",.23)], budget=1.,
        native_options={"late_second_accept":1.0})
    assert result["status"] == "budget_exhausted"
    assert np.array_equal(result["final_model"], [.25])
    assert native.counts["accept_calls"] == 3
    assert np.array_equal(native.current.model, [.25])
    assert result["adapter"]["current_accepted"]["model_sha256"] == adapter.model_sha(np.array([.25]))


def test_paid_initial_merit_after_deadline_is_not_eligible():
    result, native, _deps, _clock = run(clock=Clock(2.8), budget=3., maximum_steps=0)
    assert result["status"] == "budget_exhausted"
    assert native.counts["objective_evaluations"] == 1 and native.counts["gradient_calls"] == 0
    assert result["checkpoints"][0]["within_budget"] is False
    assert not any(c["within_budget"] for c in result["checkpoints"])


def test_prior_and_normalized_gradient_match_scalar_centered_difference():
    dependencies = adapter.load_frozen_dependencies()
    owner = adapter.ComposedObjectiveOracle(ToyNative(Clock()), initial_model=np.array([.25]),
        shape=(1,1), grid_spacing=.025, smoothing_length=.1, weight=100., prior_module=dependencies.prior)
    initial = owner.evaluate(np.array([.25]))
    owner.accept(initial)
    point = owner.evaluate(np.array([.23]))
    gradient = owner.gradient(point).gradient[0]
    data0 = .5*(.25-.1)**2

    def scalar(m):
        return .5*(m-.1)**2/data0 + 50*((.25/m)**.5-1)**2

    step = 1e-6
    assert gradient == pytest.approx((scalar(.23+step)-scalar(.23-step))/(2*step), abs=2e-8)
    assert point.objective == pytest.approx(scalar(.23), abs=2e-15)


def test_unissued_or_altered_points_fail_before_native_gradient_or_accept():
    dependencies = adapter.load_frozen_dependencies()
    native = ToyNative(Clock())
    owner = adapter.ComposedObjectiveOracle(native, initial_model=np.array([.25]), shape=(1,1),
        grid_spacing=.025, smoothing_length=.1, weight=100., prior_module=dependencies.prior)
    point = owner.evaluate(np.array([.25]))
    foreign = replace(point)
    with pytest.raises(ValueError, match="foreign or fabricated"):
        owner.accept(foreign)
    with pytest.raises(ValueError, match="foreign or fabricated"):
        owner.gradient(foreign)
    assert native.counts["gradient_calls"] == native.counts["accept_calls"] == 0
    with pytest.raises(ValueError):
        point.prior_evaluation.gradient[0,0] = 1
    object.__setattr__(point, "objective", 7.)
    with pytest.raises(ValueError, match="fingerprint changed"):
        owner.accept(point)
    assert native.counts["accept_calls"] == 0


def test_wrong_first_model_rejected_without_native_merit_query():
    dependencies = adapter.load_frozen_dependencies()
    native = ToyNative(Clock())
    owner = adapter.ComposedObjectiveOracle(native, initial_model=np.array([.25]), shape=(1,1),
        grid_spacing=.025, smoothing_length=.1, weight=0., prior_module=dependencies.prior)
    assert native.counts["evaluate_calls"] == native.counts["gradient_calls"] == 0
    with pytest.raises(ValueError, match="exact supplied initial"):
        owner.evaluate(np.array([.23]))
    assert native.counts["evaluate_calls"] == 0


def test_zero_paid_Phi0_fails_without_floor_or_prior_evaluation():
    result, native, _deps, _clock = run(native_options={"target":.25})
    assert result["status"] == "failed" and result["initial_objective"] is None
    assert native.counts["objective_evaluations"] == 1 and native.counts["gradient_calls"] == 0
    assert result["adapter"]["adapter_counts"]["prior_evaluations_completed"] == 0


def test_initial_model_is_copied_and_no_velocity_roundtrip_occurs():
    dependencies = adapter.load_frozen_dependencies()
    model = np.array([.25])
    original = model.copy()
    owner = adapter.ComposedObjectiveOracle(ToyNative(Clock()), initial_model=model, shape=(1,1),
        grid_spacing=.025, smoothing_length=.1, weight=100., prior_module=dependencies.prior)
    model[:] = .5
    point = owner.evaluate(original)
    assert point.objective == 1. and point.prior_evaluation.value == 0.
    assert owner.initial_model.tobytes() == original.tobytes()
    assert point.model.tobytes() == original.tobytes()


def test_monotone_global_anchor_rejects_reset_or_boolean_clock():
    clock = Clock(2.)
    anchor = adapter.GlobalAnchorClock(clock, 0.)
    assert anchor() == 0. and anchor() == 2.
    clock.value = 1.
    with pytest.raises(ValueError, match="globally monotone"):
        anchor()
    with pytest.raises(ValueError, match="finite true global"):
        adapter.GlobalAnchorClock(clock, True)


@pytest.mark.parametrize("weight", [0., 100.])
def test_flat_native_model_reshape_and_initial_gradient_are_consistent(weight):
    dependencies = adapter.load_frozen_dependencies()
    initial = np.array([[.2,.23,.27],[.3,.34,.38]], dtype=np.float64)
    native = ToyNative(Clock(), n=initial.size)
    owner = adapter.ComposedObjectiveOracle(native, initial_model=initial.reshape(-1), shape=(2,3),
        grid_spacing=.025, smoothing_length=.1, weight=weight, prior_module=dependencies.prior)
    point = owner.evaluate(initial.reshape(-1))
    owner.accept(point)
    actual = owner.gradient(point).gradient
    expected = (initial.reshape(-1)-.1)/(.5*np.sum((initial-.1)**2))
    assert np.array_equal(actual, expected)
    assert point.model.tobytes() == initial.tobytes() and point.objective == 1.
    if weight:
        assert point.prior_evaluation.gradient.shape == (2,3)
        assert np.array_equal(point.prior_evaluation.gradient, np.zeros((2,3)))


def test_interrupted_gradient_retains_unknown_prefix_without_new_native_queries():
    dependencies = adapter.load_frozen_dependencies()
    dependencies.engine.minimize = scripted([("accept",.23)])
    clock = Clock()
    made = []

    def factory():
        native = ToyNative(clock, interrupt_gradient=True)
        made.append(native)
        return native

    options = dependencies.engine.TrajectoryOptions(budget_seconds=100., maximum_steps=10,
        initial_step=.002, maximum_step=.02)
    with pytest.raises(KeyboardInterrupt) as caught:
        adapter.run_composed_exact(native_oracle_factory=factory,
            initial_model=np.array([.25]), shape=(1,1), grid_spacing=.025,
            smoothing_length=.1, weight=100., dependencies=dependencies, options=options,
            global_started=0., clock=clock)
    error = caught.value
    native = made[0]
    assert native.events[-1][0] == "gradient"
    prefix = error.composed_adapter_prefix
    assert prefix["adapter_counts"]["native_gradient_dispatches"] == 1
    assert prefix["adapter_counts"]["native_gradient_completed"] == 0
    assert prefix["in_flight_native_work_unknown"] is True
    assert prefix["completed_return_claim"] is False
    assert error.bounded_exact_pending_prefix["current_counter_snapshot_unknown"] is True


def test_failed_prior_construction_is_explicit_without_completed_cost_claim():
    dependencies = adapter.load_frozen_dependencies()
    options = dependencies.engine.TrajectoryOptions(budget_seconds=100., maximum_steps=0,
        initial_step=.002, maximum_step=.02)
    clock = Clock()
    native = ToyNative(clock)
    result = adapter.run_composed_exact(native_oracle_factory=lambda:native,
        initial_model=np.array([.25]), shape=(1,1), grid_spacing=0., smoothing_length=.1,
        weight=100., dependencies=dependencies, options=options, global_started=0., clock=clock)
    assert result["status"] == "failed" and result["costs_complete"] is False
    assert result["construction_prefix"] == {"native_factory_attempted": True,
        "native_factory_returned": True, "adapter_constructor_attempted": True,
        "adapter_constructor_completed": False, "creation_in_flight_costs_unknown": True}
    assert native.counts["evaluate_calls"] == native.counts["gradient_calls"] == 0


@pytest.mark.parametrize("weight", [0., 100.])
def test_actual_scipy_optimizes_manufactured_composed_objective(weight):
    result, native, _deps, _clock = run(weight=weight, actions=None, maximum_steps=30)
    assert result["status"] == "solver_success"
    assert result["final_objective"] < 1.
    assert native.counts["gradient_evaluations"] > 0
    assert native.counts["evaluate_calls"] == result["adapter"]["adapter_counts"]["native_evaluate_dispatches"]
    assert result["adapter"]["adapter_counts"]["prior_evaluations_completed"] == (native.counts["objective_evaluations"] if weight else 0)
    assert np.array_equal(native.current.model, result["final_model"])
