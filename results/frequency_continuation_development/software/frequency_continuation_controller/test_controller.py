"""Manufactured controller accounting/ownership controls; no project fixtures.

Scripted counter clocks are not timing evidence. The one native integration
uses a tiny constant model and two iterations, not a benchmark trajectory.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load("_frequency_controller_test", HERE / "controller.py")
B = load("_frequency_bounded_engine_test", HERE / "bounded_exact.py")


class ManualClock:
    def __init__(self):
        self.time = 0.

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


class ScriptedEngine:
    """Own real mock receipts; scripts only exercise controller boundary logic."""
    def __init__(self, clock, plans=()):
        self.clock, self.plans, self.calls = clock, list(plans), []

    def __call__(self, *, oracle_factory, m0, options, clock, **kwargs):
        index = len(self.calls)
        plan = self.plans[index] if index < len(self.plans) else {}
        started = clock()
        oracle = oracle_factory()
        self.calls.append(dict(deadline=options.budget_seconds, cap=options.maximum_steps,
            clock_started=started, m0=C.model_sha(m0), oracle=oracle))
        checkpoints, steps, current = [], [], C._freeze(m0)
        receipt = None

        def result(status):
            return dict(status=status, initial_model=C._freeze(m0), final_model=current,
                initial_objective=None if not checkpoints else checkpoints[0]["objective"],
                final_objective=None if receipt is None else receipt.objective,
                checkpoints=checkpoints, steps=[] if plan.get("sample_steps") else steps, costs_complete=True,
                native_callback_count=plan.get("alias_count", plan.get("callback_count", len(steps))),
                native_callback_invocations=plan.get("callback_count", len(steps)),
                native_callback_point_records=len(steps),
                total_online_costs=dict(oracle=oracle.cost_snapshot()), solver_result=None)

        if clock()-started >= options.budget_seconds:
            return result("budget_exhausted")
        self.clock.advance(plan.get("initial_extra", 0.))
        receipt = oracle.evaluate(current)
        oracle.accept(receipt)
        t = clock()-started
        checkpoints.append(dict(kind="initial", model_sha256=C.model_sha(current),
            objective=receipt.objective, within_budget=t <= options.budget_seconds,
            elapsed_seconds=t))
        if plan.get("interrupt") or plan.get("finish_failure"):
            exc = ValueError("synthetic finish failure") if plan.get("finish_failure") else KeyboardInterrupt("synthetic interruption after paid initial")
            exc.bounded_exact_pending_prefix = result("unfinalized_engine_prefix")
            raise exc
        if t <= options.budget_seconds and options.maximum_steps:
            action = plan.get("action", "move")
            candidate = current if action == "equal" else C._freeze(current-.01)
            proposed = oracle.evaluate(candidate)
            if action == "wrong_current":
                oracle.accept(proposed)  # Keep final/checkpoint old: must fail closed.
            else:
                strict = proposed.objective < receipt.objective
                steps.append(dict(native_callback=dict(strict_decrease_endpoint_eligible=strict)))
                if strict:
                    oracle.accept(proposed)
                    receipt, current = proposed, candidate
                    checkpoints.append(dict(kind="accepted", model_sha256=C.model_sha(current),
                        objective=receipt.objective, within_budget=True, elapsed_seconds=clock()-started))
        self.clock.advance(plan.get("finish_tail", 0.))
        return result(plan.get("status", "solver_terminated") if t <= options.budget_seconds else "budget_exhausted")


def manufactured(clock, plans=(), *, prefix=.2, bad_counter=False):
    created = []

    class Physics:
        def __init__(self, **config):
            self.n = math.prod(config["shape"])

    class Oracle:
        def __init__(self, physics, blocks, receivers, **kwargs):
            self.n, self.blocks = physics.n, tuple(blocks)
            self.physics = physics
            self.current, self.cache = None, {}
            self.counts = dict(queries=0, hits=0, accepts=0)
            self.contract = dict(geometry_sha256=str([b.frequency_hz for b in blocks]),
                data_sha256=str([b.observed.tolist() for b in blocks]))
            created.append(self)
            clock.advance(.05)

        def evaluate(self, model):
            key = C.model_sha(model)
            if key in self.cache:
                self.counts["hits"] += 1
                clock.advance(.01)
                return self.cache[key]
            self.counts["queries"] += 1
            clock.advance(.05)
            target = .1 if len(self.blocks) == 8 else .3
            phi = float(np.sum((model-target)**2))*len(self.blocks)
            header = dict(objective=phi, model_sha256=key, **self.contract)
            receipt = SimpleNamespace(owner=self, model=C._freeze(model), objective=phi,
                to_dict=lambda header=header: dict(header))
            self.cache[key] = receipt
            return receipt

        def accept(self, receipt):
            if receipt.owner is not self:
                raise ValueError("foreign manufactured receipt")
            self.counts["accepts"] += 1
            clock.advance(.01)
            self.current = receipt

        def gradient(self, receipt):
            if receipt.owner is not self:
                raise ValueError("foreign manufactured receipt")
            target = .1 if len(self.blocks) == 8 else .3
            return SimpleNamespace(gradient=2*len(self.blocks)*(receipt.model-target))

        def cost_snapshot(self):
            return dict(physics={}, oracle=dict(self.counts))

    engine = ScriptedEngine(clock, plans)
    api = C.NativeAPI(Physics, lambda s, f, y, label: SimpleNamespace(
        source_index=s, frequency_hz=f, observed=y, label=label), Oracle,
        B.TrajectoryOptions, engine, Path("/manufactured"), dict(kind="manufactured_only"))

    def prepare(context):
        counts = dict(loads=True if bad_counter else 1, observation_blocks=8)
        context.register_preparation_costs("real_executed_mock_preparation", lambda: dict(counts))
        clock.advance(prefix)
        blocks = tuple(api.block_type(0, f, np.array([1+0j]), str(f)) for f in C.ALL8)
        return C.PreparedProblem(dict(shape=(2, 2), spacing=.05, damping_width=0,
            damping_strength=1., factorization_cache_size=0), np.full(4, .4), blocks, (1,))

    return api, prepare, created, engine


def run(arm="joint8", *, plans=(), prefix=.2, progress=None, bad_counter=False):
    clock = ManualClock()
    api, prepare, created, engine = manufactured(clock, plans, prefix=prefix, bad_counter=bad_counter)
    attempt = {}
    result = C.measure_controller_return(arm=arm, prepare=prepare, api=api, clock=clock,
        settings=C.ControllerSettings(single_stage_maximum_steps=2, per_stage_maximum_steps=1),
        attempt=attempt, progress=None if progress is None else lambda row: progress(clock, row))
    return result, created, engine, attempt


@pytest.mark.parametrize("arm,owner_count", [("joint8", 1), ("low4_then8", 3), ("restart8", 2), ("original4", 2)])
def test_four_arms_owned_surveys_common_objectives_and_absolute_caps(arm, owner_count):
    result, owners, engine, _ = run(arm)
    p = result["payload"]
    assert result["runtime_gate"]["eligible"] and len(owners) == len(p["owners"]) == owner_count
    assert len({id(o) for o in owners}) == owner_count
    assert len({id(o.physics) for o in owners}) == owner_count
    assert [tuple(s["frequencies_hz"]) for s in p["stages"]] == list(C.ARMS[arm])
    assert [c["deadline"] for c in engine.calls] == ([3.75, 7.5] if len(C.ARMS[arm]) == 2 else [7.5])
    assert [c["cap"] for c in engine.calls] == ([1, 1] if len(C.ARMS[arm]) == 2 else [2])
    assert all(c["clock_started"] == 0 for c in engine.calls)
    assert p["common_initial_full8"]["objective"] == pytest.approx(8*4*(.4-.1)**2)
    assert p["common_final_full8"]["objective"] == pytest.approx(8*float(np.sum((p["final_model"]-.1)**2)))
    assert p["common_full8_ratio"] == p["common_final_full8"]["objective"]/p["common_initial_full8"]["objective"]
    expected_queries = sum(o.counts["queries"] for o in owners)
    assert p["owner_costs_summed_once"]["native"]["oracle"]["queries"] == expected_queries
    assert p["preparation_costs_separate"]["observation_blocks"] == 8
    if arm == "original4":
        stage = p["stages"][0]["engine_result"]
        assert stage["final_objective"]/stage["initial_objective"] != p["common_full8_ratio"]
    assert all("initial_model" not in s["engine_result"] and "final_model" not in s["engine_result"] for s in p["stages"])
    json.dumps(C._jsonable(p), allow_nan=False)


def test_prefix_beyond_deadline_has_no_paid_initial_fallback():
    result, owners, engine, _ = run("low4_then8", prefix=4.)
    assert not owners and not engine.calls
    assert result["payload"]["final_model"] is None and not result["runtime_gate"]["eligible"]
    assert result["payload"]["stages"][1]["status"] == "unattempted_no_prior_paid_state"


def test_late_initial_not_eligible_and_separate_baseline_cannot_seed_stage2():
    result, _, engine, _ = run("low4_then8", plans=[dict(initial_extra=4.)])
    assert len(engine.calls) == 1 and result["payload"]["common_initial_full8"] is not None
    assert result["payload"]["stages"][0]["eligible_checkpoint_count"] == 0
    assert result["payload"]["final_model"] is None and not result["runtime_gate"]["eligible"]


def test_stage2_late_initial_returns_only_previous_paid_model_with_paid_full8_metric():
    result, _, _, _ = run("restart8", plans=[dict(finish_tail=3.3), dict(initial_extra=3.9)])
    p = result["payload"]
    assert p["stages"][1]["eligible_checkpoint_count"] == 0
    assert p["final_model_origin"]["stage_index"] == 0
    assert np.array_equal(p["final_model"], np.full(4, .39))
    assert p["common_final_full8"] is not None and result["runtime_gate"]["eligible"]


def test_equal_callback_counts_without_extra_checkpoint_and_early_handoff():
    result, _, engine, _ = run("restart8", plans=[dict(action="equal"), dict(action="equal")])
    assert result["runtime_gate"]["eligible"]
    assert result["payload"]["stages"][1]["call_start_age_seconds"] < 3.75
    assert len(engine.calls) == 2
    assert all(s["native_callback_events"] == 1 and s["eligible_checkpoint_count"] == 1 for s in result["payload"]["stages"])


def test_aggregate_callback_count_not_sampled_steps_controls_cap_and_reporting():
    result, _, _, _ = run(plans=[dict(sample_steps=True)])
    stage = result["payload"]["stages"][0]
    assert stage["retained_native_callback_events"] == 0 and stage["native_callback_events"] == 1
    assert result["runtime_gate"]["eligible"]
    exceeded, _, _, _ = run(plans=[dict(sample_steps=True, callback_count=3)])
    assert not exceeded["runtime_gate"]["eligible"]
    assert "native callback safety cap exceeded" in exceeded["payload"]["error"]


def test_boolean_aggregate_callback_count_is_not_work():
    result, _, _, _ = run(plans=[dict(callback_count=True)])
    assert not result["runtime_gate"]["eligible"]
    assert "callback count must be a nonnegative integer" in result["payload"]["error"]


def test_boolean_callback_alias_is_rejected_even_when_numeric_equality_holds():
    result, _, _, _ = run(plans=[dict(alias_count=True)])
    assert not result["runtime_gate"]["eligible"]
    assert "invalid engine callback point count or invocation alias" in result["payload"]["error"]


def test_actual_finish_tail_is_paid_and_late_api_return_censored():
    def late_tail(clock, event):
        if event["phase"] == "finalizing_payload":
            clock.advance(8.)
    result, _, _, _ = run(progress=late_tail)
    assert result["payload"]["method_valid_before_return_gate"] is True
    assert not result["runtime_gate"]["eligible"] and result["runtime_gate"]["event_seconds"] is None
    assert result["timing"]["actual_api_return_seconds"] > 8


def test_regressing_return_clock_cannot_credit_a_positive_but_earlier_stamp():
    def regress(clock, event):
        if event["phase"] == "finalizing_payload":
            clock.advance(-.1)
    result, _, _, _ = run(progress=regress)
    assert result["timing"]["actual_api_return_seconds"] >= 0
    assert result["timing"]["return_stamp_monotone"] is False
    assert not result["runtime_gate"]["eligible"]


def test_wrong_current_not_repaired_by_final_metric_and_failed_chain_keeps_last_paid_model():
    broken, _, _, _ = run(plans=[dict(action="wrong_current")])
    assert not broken["runtime_gate"]["eligible"]
    assert "authentic current receipt" in broken["payload"]["error"]
    failed, _, engine, _ = run("restart8", plans=[dict(status="failed")])
    assert len(engine.calls) == 1 and not failed["runtime_gate"]["eligible"]
    assert failed["payload"]["final_model"] is not None


def test_bad_counter_is_unknown_not_zero_or_eligible():
    result, _, _, _ = run(bad_counter=True)
    p = result["payload"]
    assert not p["costs_complete"] and not result["runtime_gate"]["eligible"]
    assert p["preparation_costs"]["real_executed_mock_preparation"]["latest_counts"] is None
    assert p["preparation_costs_separate"] is None


def test_failed_constructor_retains_attempt_unknown_cost_and_all_stage_slots():
    clock, attempt = ManualClock(), {}
    api, prepare, _, _ = manufactured(clock)
    class BrokenPhysics:
        def __init__(self, **kwargs):
            clock.advance(.03)
            raise RuntimeError("synthetic constructor failure")
    api = C.NativeAPI(BrokenPhysics, api.block_type, api.oracle_type, api.options_type,
        api.engine_run, api.package_root, api.source_binding)
    result = C.measure_controller_return(arm="restart8", prepare=prepare, api=api,
        clock=clock, attempt=attempt)
    p = result["payload"]
    assert len(p["stages"]) == 2 and len(p["owners"]) == 1
    assert p["owners"][0]["status"] == "construction_failed"
    assert p["owners"][0]["in_flight_creation_costs_unknown"]
    assert p["owner_costs_summed_once"] is None and not p["costs_complete"]
    assert not result["runtime_gate"]["eligible"]


def test_reused_native_owner_identity_is_rejected(monkeypatch):
    clock, attempt = ManualClock(), {}
    api, prepare, created, _ = manufactured(clock)
    def reused_new(cls, *args, **kwargs):
        return created[0] if created else object.__new__(cls)
    monkeypatch.setattr(api.oracle_type, "__new__", staticmethod(reused_new))
    result = C.measure_controller_return(arm="restart8", prepare=prepare, api=api,
        clock=clock, attempt=attempt)
    assert not result["runtime_gate"]["eligible"]
    assert "new distinct native owner" in result["payload"]["error"]
    assert result["payload"]["owner_costs_summed_once"] is None


def test_preparation_interrupt_preserves_unknown_work_and_all_stage_slots():
    clock, attempt = ManualClock(), {}
    api, _, _, _ = manufactured(clock)
    def interrupted(context):
        context.register_preparation_costs("partial", lambda: dict(loads=1))
        raise KeyboardInterrupt("synthetic preparation interrupted")
    with pytest.raises(KeyboardInterrupt):
        C.measure_controller_return(arm="restart8", prepare=interrupted, api=api, clock=clock, attempt=attempt)
    assert len(attempt["stages"]) == 2 and attempt["payload"]["preparation_in_flight_counts_unknown"]
    assert attempt["global_started_perf_counter"] == 0.
    assert not attempt["payload"]["costs_complete"]
    assert attempt["preparation_costs"]["partial"]["latest_counts"]["loads"] == 1
    assert attempt["payload"]["preparation_costs_separate"] is None


def test_engine_interrupt_prefix_retained_without_credit():
    clock, attempt = ManualClock(), {}
    api, prepare, _, _ = manufactured(clock, plans=[dict(interrupt=True)])
    with pytest.raises(KeyboardInterrupt):
        C.measure_controller_return(arm="restart8", prepare=prepare, api=api, clock=clock, attempt=attempt)
    assert attempt["payload"]["status"] == "interrupted"
    assert attempt["payload"]["final_model"] is not None
    assert attempt["stages"][0]["engine_result"]["status"] == "unfinalized_engine_prefix"
    assert attempt["stages"][0]["engine_prefix_finalized"] is False
    assert attempt["stages"][0]["engine_prefix_is_completed_return"] is False
    assert not attempt["payload"]["method_valid_before_return_gate"]


def test_ordinary_engine_finish_failure_also_retains_unfinalized_prefix():
    result, _, _, _ = run(plans=[dict(finish_failure=True)])
    p = result["payload"]
    assert p["status"] == "failed_controller" and not result["runtime_gate"]["eligible"]
    assert p["final_model"] is not None and p["in_flight_work_unknown"] is True
    assert p["stages"][0]["engine_prefix_finalized"] is False
    assert p["stages"][0]["engine_prefix_is_completed_return"] is False
    assert not p["costs_complete"]


@pytest.mark.parametrize("arm", list(C.ARMS))
def test_real_tiny_native_oracle_all_four_arms_with_bounded_engine(arm):
    repo = HERE.parents[1] / "wavecert-fwi"
    api = C.load_native_api(repo, engine_run=B.run_exact_lbfgsb,
        engine_source_path=HERE / "bounded_exact.py", engine_source_sha256=C.file_sha(HERE / "bounded_exact.py"))
    clock = ManualClock()  # Frozen correctness clock; no measured performance claim.
    def prepare(context):
        config = dict(shape=(4, 5), spacing=.05, damping_width=1, damping_strength=4., factorization_cache_size=0)
        physics = api.physics_type(**config)
        context.register_preparation_costs("observations", physics.cost_snapshot)
        truth, blocks, receivers = np.full(physics.n, .22), [], (1, 8, 17)
        for f in C.ALL8:
            for source in (6, 13):
                y = physics.solve_state(truth, physics.source(source), f)[list(receivers)]
                blocks.append(api.block_type(source, f, y, f"{source}@{f}"))
        return C.PreparedProblem(config, np.full(physics.n, .24), tuple(blocks), receivers)
    result = C.measure_controller_return(arm=arm, prepare=prepare, api=api,
        settings=C.ControllerSettings(single_stage_maximum_steps=2, per_stage_maximum_steps=2), clock=clock)
    assert result["runtime_gate"]["eligible"], result["payload"].get("error")
    assert result["payload"]["source_binding"]["production_sha256"] == C.PRODUCTION_SHA256
    assert result["payload"]["common_final_full8"]["objective"] >= 0
    assert all(s["engine_result"]["checkpoint_vectors_retained"] is False for s in result["payload"]["stages"])
    assert all(s["native_callback_events"] <= 2 for s in result["payload"]["stages"])
