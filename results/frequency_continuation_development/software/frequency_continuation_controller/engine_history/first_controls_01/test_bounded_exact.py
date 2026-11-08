"""Manufactured equivalence controls, never project data or runtime evidence.

Scripted solvers replace bindings only in the staged engine and a private
validation copy of the frozen old source. Production modules stay untouched.
One tiny real Helmholtz control uses the actual SciPy optimizer and exact oracle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bounded_exact as bounded

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]/"wavecert-fwi"
OLD_PATH = REPO/"src/wavecert/optimization/paid_descent.py"
OLD_SHA = "b863af87f03c2470fde8d72412b1c89e76e389a528cb7d8b08efe89ceeb3a12c"


def load_old():
    assert hashlib.sha256(OLD_PATH.read_bytes()).hexdigest() == OLD_SHA
    name = "_bounded_exact_private_old_control"
    spec = importlib.util.spec_from_file_location(name, OLD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Clock:
    """Explicit synthetic work units, never measured timings."""

    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, value):
        self.value += value


class Oracle:
    def __init__(self, clock, target, *, setup=.1, merit=.2, gradient=.3,
                 accept_second=0., fail_gradient=False):
        self.clock, self.target = clock, np.asarray(target)
        self.merit_time, self.gradient_time, self.accept_second = merit, gradient, accept_second
        self.fail_gradient = fail_gradient
        self.current = self.trial = None
        self.events, self.accepted_models = [], []
        self.counts = dict(constructions=1, merits=0, merit_hits=0, gradients=0, gradient_hits=0, accepts=0)
        clock.advance(setup)

    def evaluate(self, model):
        model = np.asarray(model, dtype=np.float64)
        self.events.append(("evaluate", bounded._digest(model)))
        for receipt in (self.current, self.trial):
            if receipt is not None and np.array_equal(receipt.model, model):
                self.counts["merit_hits"] += 1
                return receipt
        self.counts["merits"] += 1
        self.clock.advance(self.merit_time)
        receipt = SimpleNamespace(model=bounded._immutable(model),
            objective=.5*float(np.sum((model-self.target)**2)), owner=self, g=None)
        self.trial = receipt
        return receipt

    def gradient(self, receipt):
        assert receipt.owner is self
        self.events.append(("gradient", bounded._digest(receipt.model)))
        if receipt.g is None:
            self.counts["gradients"] += 1
            self.clock.advance(self.gradient_time)
            if self.fail_gradient:
                raise RuntimeError("manufactured gradient failure")
            receipt.g = receipt.model-self.target
        else:
            self.counts["gradient_hits"] += 1
        return SimpleNamespace(gradient=receipt.g, receipt=receipt)

    def accept(self, receipt):
        assert receipt.owner is self
        self.events.append(("accept", bounded._digest(receipt.model)))
        self.counts["accepts"] += 1
        if self.counts["accepts"] == 2:
            self.clock.advance(self.accept_second)
        self.current, self.trial = receipt, None
        self.accepted_models.append(receipt.model.copy())

    def cost_snapshot(self):
        self.events.append(("cost_snapshot",))
        return dict(self.counts)


def options(**changes):
    return replace(bounded.TrajectoryOptions(budget_seconds=100., maximum_steps=2,
        initial_step=.002, maximum_step=.02), **changes)


def scripted(candidates, *, gradient=True):
    """A native callable-contract fixture, not an optimization algorithm."""
    def solve(function, initial, *, jac, method, bounds, callback, options):
        assert method == "L-BFGS-B"
        assert set(options) == {"maxiter", "maxls", "maxcor", "ftol", "gtol"}
        assert len(bounds) == len(initial)
        for model in candidates[:options["maxiter"]]:
            function(model)
            if gradient:
                jac(model)
            callback(model)
        count = min(len(candidates), options["maxiter"])
        return SimpleNamespace(success=True, status=0, message="manufactured native return",
                               nit=count, nfev=count, njev=count if gradient else 0)
    return solve


def normalized(result):
    """Only the documented representation additions/removals are excluded."""
    extra = {"initial_model", "checkpoint_vectors_retained", "native_callback_count",
             "maximum_steps_safety_cap", "scipy_maxfun_default", "scipy_maxfun_passed_explicitly"}
    value = {k: v for k, v in result.items() if k not in extra}
    value["checkpoints"] = [{k: v for k, v in checkpoint.items() if k != "model"}
                            for checkpoint in result["checkpoints"]]
    return bounded._strict_metadata(value)


def paired_toy(candidates, *, initial=(.8,), target=(.3,), opts=None, gradient=True, **oracle_options):
    old = load_old()
    results, owners = [], []
    original_bounded_solver = bounded.minimize
    try:
        for module in (old, bounded):
            module.minimize = scripted([np.asarray(v, dtype=float) for v in candidates], gradient=gradient)
            clock, made = Clock(), []
            def factory(clock=clock, made=made):
                owner = Oracle(clock, target, **oracle_options)
                made.append(owner)
                return owner
            result = module.run_exact_lbfgsb(oracle_factory=factory, m0=np.asarray(initial),
                lower_bound=.1, upper_bound=1., options=opts or options(), clock=clock)
            results.append(result)
            owners.append(made[0])
    finally:
        bounded.minimize = original_bounded_solver
    assert owners[0].events == owners[1].events
    assert owners[0].counts == owners[1].counts
    assert [bounded._digest(v) for v in owners[0].accepted_models] == [bounded._digest(v) for v in owners[1].accepted_models]
    assert normalized(results[0]) == normalized(results[1])
    return results, owners


def test_strict_native_callbacks_and_paid_requests_match_frozen_old_engine():
    results, owners = paired_toy([(.6,), (.4,)])
    old, new = results
    assert new["status"] == old["status"] == "solver_success"
    assert new["native_callback_count"] == 2 and len(new["checkpoints"]) == 3
    assert np.array_equal(new["final_model"], [.4])
    assert owners[1].current.owner is owners[1]
    assert all("model" not in row for row in new["checkpoints"])


def test_equal_native_callback_is_paid_without_endpoint_credit():
    results, owners = paired_toy([(.3, .8)], initial=(.8, .3), target=(0., 0.))
    new = results[1]
    assert new["native_callback_count"] == 1 and len(new["checkpoints"]) == 1
    assert not new["steps"][0]["accepted"]
    assert not new["steps"][0]["native_callback"]["strict_decrease_endpoint_eligible"]
    assert owners[1].counts["merits"] == 2 and owners[1].counts["gradients"] == 1
    assert np.array_equal(new["final_model"], [.8, .3])


@pytest.mark.parametrize("scenario", ["setup", "exact_initial_boundary", "late_initial", "late_gradient", "late_commit", "gradient_failure"])
def test_clock_guards_and_failed_prefixes_match_frozen_old_engine(scenario):
    kw = dict(opts=options(budget_seconds=1.))
    if scenario == "setup":
        kw.update(setup=1.)
    elif scenario == "exact_initial_boundary":
        kw.update(setup=.25, merit=.75)
    elif scenario == "late_initial":
        kw.update(setup=.25, merit=1.)
    elif scenario == "late_gradient":
        kw.update(setup=.1, merit=.3, gradient=.7)
    elif scenario == "late_commit":
        kw.update(setup=.1, merit=.1, gradient=.1, accept_second=1.)
    else:
        kw.update(fail_gradient=True, opts=options())
    results, owners = paired_toy([(.6,)], **kw)
    new = results[1]
    assert new["status"] == ("failed" if scenario == "gradient_failure" else "budget_exhausted")
    assert np.array_equal(new["final_model"], [.8])
    if scenario == "late_initial":
        assert len(new["checkpoints"]) == 1 and not new["checkpoints"][0]["within_budget"]
    if scenario == "late_commit":
        assert owners[1].counts["accepts"] == 3
        assert np.array_equal(owners[1].current.model, [.8])
        assert len(new["checkpoints"]) == 1 and not new["steps"][0]["accepted"]


def test_zero_steps_pays_initial_merit_without_invoking_native_solver():
    results, owners = paired_toy([], opts=options(maximum_steps=0))
    assert results[1]["status"] == "maximum_steps"
    assert owners[1].counts["merits"] == owners[1].counts["accepts"] == 1


def test_many_checkpoints_retain_only_two_top_level_model_vectors():
    initial = np.full(64, .9)
    candidates = [np.full(64, .9-(i+1)*.001) for i in range(128)]
    results, _ = paired_toy(candidates, initial=initial, target=np.full(64, .3),
        opts=options(maximum_steps=4096, budget_seconds=1000.))
    new = results[1]
    assert len(new["checkpoints"]) == 129 and new["native_callback_count"] == 128
    assert new["maximum_steps_safety_cap"] == 4096
    assert new["scipy_maxfun_default"] == 15000 and new["scipy_maxfun_passed_explicitly"] is False
    def arrays(value):
        if isinstance(value, np.ndarray):
            return [value]
        if isinstance(value, dict):
            return [a for item in value.values() for a in arrays(item)]
        if isinstance(value, (list, tuple)):
            return [a for item in value for a in arrays(item)]
        return []
    assert arrays(new["checkpoints"]) == arrays(new["steps"]) == []
    assert len(arrays(new)) == 2
    assert sum(a.nbytes for a in arrays(new)) == 2*initial.nbytes
    with pytest.raises(ValueError):
        new["initial_model"].setflags(write=True)
    with pytest.raises(ValueError):
        new["final_model"].setflags(write=True)
    json.dumps({k: v for k, v in new.items() if k not in {"initial_model", "final_model"}}, allow_nan=False)


def test_interruption_propagates_with_paid_bounded_prefix():
    old, requested = load_old(), []
    original_bounded_solver = bounded.minimize
    def interrupted_solver(function, initial, **kwargs):
        function(np.full_like(initial, .6))
        raise KeyboardInterrupt("manufactured native interruption")
    try:
        for module in (old, bounded):
            module.minimize = interrupted_solver
            clock, made = Clock(), []
            def factory(clock=clock, made=made):
                owner = Oracle(clock, [.3])
                made.append(owner)
                return owner
            with pytest.raises(KeyboardInterrupt, match="manufactured") as caught:
                module.run_exact_lbfgsb(oracle_factory=factory, m0=np.array([.8]),
                    lower_bound=.1, upper_bound=1., options=options(), clock=clock)
            requested.append([v for v in made[0].events if v[0] != "cost_snapshot"])
            if module is bounded:
                prefix = caught.value.bounded_exact_prefix
                assert prefix["status"] == "interrupted" and prefix["costs_complete"]
                assert prefix["total_online_costs"]["oracle"]["merits"] == 2
                assert np.array_equal(prefix["final_model"], [.8])
                assert prefix["steps"][-1]["termination"] == "interrupted"
    finally:
        bounded.minimize = original_bounded_solver
    assert requested[0] == requested[1]


def test_tiny_real_helmholtz_oracle_matches_requests_receipts_and_native_result():
    import wavecert
    from wavecert.physics.exact_merit import ExactMeritOracle
    from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock

    package = REPO/"src/wavecert"
    assert Path(wavecert.__file__).resolve().parent == package
    physics = Helmholtz2D((4, 5), spacing=.05, damping_width=1, damping_strength=4.)
    truth = np.linspace(.2, .23, physics.n)
    receivers, blocks = [1, 7, 7, 14], []
    for frequency in (2.3, 3.1):
        for source in (3, 11):
            observed = physics.solve_state(truth, physics.source(source), frequency)[receivers]
            blocks.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
    class TraceOwner:
        def __init__(self):
            self.owner = ExactMeritOracle(physics, blocks, receivers, reference_package_root=package)
            self.events = []
        def evaluate(self, model):
            self.events.append(("evaluate", bounded._digest(model)))
            return self.owner.evaluate(model)
        def gradient(self, receipt):
            self.events.append(("gradient", receipt.model_sha256))
            return self.owner.gradient(receipt)
        def accept(self, receipt):
            self.events.append(("accept", receipt.model_sha256))
            self.owner.accept(receipt)
        def cost_snapshot(self):
            self.events.append(("cost_snapshot",))
            return self.owner.cost_snapshot()
    results, made = [], []
    for module in (load_old(), bounded):
        def factory():
            owner = TraceOwner()
            made.append(owner)
            return owner
        results.append(module.run_exact_lbfgsb(oracle_factory=factory, m0=np.full(physics.n, .25),
            lower_bound=.12, upper_bound=.4, options=options(maximum_steps=2), clock=lambda: 0.))
    assert normalized(results[0]) == normalized(results[1])
    assert made[0].events == made[1].events
    assert made[0].owner.cost_snapshot() == made[1].owner.cost_snapshot()
    assert results[1]["solver_result"] is not None
    assert results[1]["solver_result"]["nit"] <= 2
    assert len(results[1]["checkpoints"]) >= 2
