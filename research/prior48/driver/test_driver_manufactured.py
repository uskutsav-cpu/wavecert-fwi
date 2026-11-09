"""Tiny scalar objectives only; no project array decoding, native owner or PDE."""
from __future__ import annotations

import copy
import gzip
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import composed_exact_adapter as adapter
import run_prior48 as driver


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, amount):
        self.value += amount


@dataclass(eq=False)
class ScalarPoint:
    model: np.ndarray
    objective: float
    owner: object

    def to_dict(self):
        return {"kind": "manufactured_scalar_not_native_receipt", "objective": self.objective,
            "model_sha256": adapter.model_sha(self.model), "geometry_sha256": "synthetic", "data_sha256": "synthetic"}


class ScalarNative:
    def __init__(self, clock, *, late_final=0., late_accept=0., interrupt=False):
        self.clock, self.n = clock, 1
        self.current = self.trial = None
        self.issued, self.accepted, self.events = set(), [], []
        self.counts = {"evaluate": 0, "gradient": 0, "accept": 0, "merit": 0}
        self.late_final, self.late_accept, self.interrupt = late_final, late_accept, interrupt
        clock.advance(.1)

    def evaluate(self, model):
        self.counts["evaluate"] += 1
        self.events.append("evaluate")
        if self.counts["evaluate"] == 2:
            self.clock.advance(self.late_final)
        for point in (self.current, self.trial):
            if point is not None and np.array_equal(point.model, model):
                return point
        self.clock.advance(.2)
        self.counts["merit"] += 1
        point = ScalarPoint(adapter.immutable(model), .5*float((model[0]-.1)**2), self)
        self.issued.add(point)
        self.trial = point
        return point

    def gradient(self, point):
        assert point in self.issued and point.owner is self
        self.counts["gradient"] += 1
        self.events.append("gradient")
        self.clock.advance(.2)
        if self.interrupt:
            raise KeyboardInterrupt("manufactured in-flight work")
        return SimpleNamespace(gradient=adapter.immutable(point.model-.1), receipt=point)

    def accept(self, point):
        assert point in self.issued and point.owner is self
        self.counts["accept"] += 1
        self.events.append("accept")
        if self.counts["accept"] == 2:
            self.clock.advance(self.late_accept)
        self.current, self.trial = point, None
        self.accepted.append(point.model.copy())

    def cost_snapshot(self):
        self.events.append("snapshot")
        return {"manufactured": dict(self.counts)}


def scripted(points):
    def minimize(function, initial, *, jac, method, bounds, callback, options):
        for value in points:
            model = np.array([value], dtype=np.float64)
            function(model)
            jac(model)
            callback(model)
        return SimpleNamespace(success=True, status=0, message="manufactured scalar sequence",
            nit=len(points), nfev=len(points), njev=len(points))
    return minimize


def scalar_run(*, weight=100., points=(), maximum_steps=10, preparation_time=.2, native_options=None,
               serializer_time=0., monkeypatch=None, attempt_out=None):
    p, _ = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    p = copy.deepcopy(p)
    p["physics"]["shape"] = [1, 1]
    p["maximum_steps"] = maximum_steps
    deps = adapter.load_frozen_dependencies()
    deps.engine.minimize = scripted(points)  # Explicit manufactured call sequence only.
    runtime = SimpleNamespace(adapter=adapter, dependencies=deps, np=np)
    clock, made = Clock(), []
    attempt = {} if attempt_out is None else attempt_out

    def prepare():
        clock.advance(preparation_time)
        return SimpleNamespace(initial=np.array([.25]), blocks=(), receivers=(), physics={})

    def make_native(problem):
        assert not hasattr(problem, "truth")
        owner = ScalarNative(clock, **(native_options or {}))
        made.append(owner)
        return owner

    if serializer_time:
        original = driver.encoded
        def paid_serialization(value):
            clock.advance(serializer_time)
            return original(value)
        monkeypatch.setattr(driver, "encoded", paid_serialization)
    measurement = driver.execute_policy(prepare=prepare, make_native=make_native,
        runtime=runtime, p=p, weight=weight, check_static=lambda: clock.advance(.1),
        attempt=attempt, clock=clock)
    return measurement, made[0], attempt, clock, runtime, p


def test_default_is_metadata_only_and_full48_order_is_already_bound(monkeypatch, capsys):
    monkeypatch.setattr(driver, "load_runtime", lambda *a: pytest.fail("project runtime entered"))
    assert driver.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "metadata_only_no_project_arrays_or_PDE"
    assert value["planned_runs"] == 48
    p, _ = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    rows = driver.schedule(p)
    assert p["geometry"]["frequencies_hz"] == driver.FREQUENCIES
    assert p["geometry"]["blocks_per_model"] == 48
    assert p["original_geometry_context"]["blocks_per_model"] == 24
    assert rows == json.loads(Path(driver.__file__).with_name("FROZEN48_SCHEDULE.json").read_text())
    assert len({(r["case_index"], r["repetition"], r["method"]) for r in rows}) == 48
    assert all(r["status"] == "unattempted" for r in rows)


@pytest.mark.parametrize("field", ["authorization", "driver_sha256", "protocol_sha256", "schedule_sha256", "independent_source_review_clear"])
def test_execution_requires_exact_separate_root_go(field):
    p, _ = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    go = {"authorization": "GO_CLASSICAL_PRIOR48_DEVELOPMENT", "driver_sha256": "driver", "protocol_sha256": "protocol",
        "schedule_sha256": p["schedule_sha256"], "production_source_sha256": driver.PRODUCTION, "independent_source_review_clear": True}
    driver.validate_go(go, p, "driver", "protocol")
    go[field] = "changed"
    with pytest.raises(driver.StaleFreeze):
        driver.validate_go(go, p, "driver", "protocol")


def test_self_rehashed_schedule_tamper_still_fails_fixed_order(tmp_path):
    source = Path(driver.__file__).parent
    p, _ = driver.load_protocol(source / "protocol.json")
    for name in ["protocol.json", *p["local_dependency_sha256"]]:
        shutil.copyfile(source/name, tmp_path/name)
    rows = json.loads((tmp_path/"FROZEN48_SCHEDULE.json").read_text())
    rows[0]["weight"] = 999
    (tmp_path/"FROZEN48_SCHEDULE.json").write_bytes(driver.encoded(rows))
    p["schedule_sha256"] = p["local_dependency_sha256"]["FROZEN48_SCHEDULE.json"] = driver.sha(tmp_path/"FROZEN48_SCHEDULE.json")
    (tmp_path/"protocol.json").write_bytes(driver.encoded(p))
    with pytest.raises(driver.StaleFreeze, match="slot order"):
        driver.load_protocol(tmp_path/"protocol.json")


@pytest.mark.parametrize("weight", [0., 100.])
def test_initial_Phi0_and_final_query_are_paid_without_roundtrip(weight):
    measurement, owner, _attempt, _clock, _runtime, _p = scalar_run(weight=weight, maximum_steps=0)
    payload = measurement["payload"]
    assert payload["method_valid"] is True
    assert owner.counts["merit"] == 1 and owner.counts["evaluate"] == 2
    assert owner.counts["gradient"] == 0
    assert measurement["models"]["initial_model"].tobytes() == np.array([.25]).tobytes()
    assert payload["engine"]["adapter"]["initial_paid_Phi0"] == .5*(.25-.1)**2
    assert payload["metric_tail_counts"]["prior_evaluations_completed"] == int(weight > 0)
    assert payload["engine"]["adapter"]["adapter_counts"]["prior_evaluations_completed"] == int(weight > 0)
    assert json.loads(measurement["serialized"])["final_model"] == [.25]
    assert payload["final_F"] == 1.


def test_raw_Phi_increase_with_F_decrease_remains_valid():
    m, owner, *_ = scalar_run(points=(.21, .23))
    assert m["payload"]["method_valid"] is True
    assert m["payload"]["final_F"] < 1.
    assert np.array_equal(m["models"]["final_model"], [.23])
    assert .5*(owner.accepted[-1][0]-.1)**2 > .5*(owner.accepted[-2][0]-.1)**2
    baseline, *_ = scalar_run(weight=0., points=(.21, .23))
    assert baseline["payload"]["engine"]["status"] == "failed"
    assert baseline["payload"]["method_valid"] is False
    assert driver.return_gate(baseline, baseline["inner_return_age"], 8.)["eligible"] is False


@pytest.mark.parametrize("tail", ["final_metric", "serializer"])
def test_complete_return_charges_final_metrics_and_serialization(tail, monkeypatch):
    kwargs = {"maximum_steps": 0, "monkeypatch": monkeypatch}
    if tail == "final_metric":
        kwargs["native_options"] = {"late_final": 8.}
    else:
        kwargs["serializer_time"] = 8.
    measurement, _owner, _attempt, clock, *_ = scalar_run(**kwargs)
    assert measurement["payload"]["method_valid"] is True
    assert driver.return_gate(measurement, clock(), 8.)["eligible"] is False


def test_preparation_is_inside_global_engine_deadline_not_a_new_local_budget():
    measurement, owner, *_ = scalar_run(preparation_time=7.6)
    assert measurement["payload"]["engine"]["status"] == "budget_exhausted"
    assert measurement["payload"]["method_valid"] is False
    assert owner.counts["evaluate"] == owner.counts["gradient"] == 0


def test_late_native_promotion_restores_last_eligible_owned_model():
    m, owner, *_ = scalar_run(points=(.23,), native_options={"late_accept": 7.0})
    assert m["payload"]["engine"]["status"] == "budget_exhausted"
    assert np.array_equal(m["models"]["final_model"], [.25])
    assert np.array_equal(owner.current.model, [.25])
    assert owner.counts["accept"] == 3
    assert m["payload"]["final_raw_native_header"]["model_sha256"] == adapter.model_sha(np.array([.25]))


def test_failed_preparation_marks_unknown_and_never_creates_owner():
    attempt, clock = {}, Clock()
    def prepare():
        raise RuntimeError("manufactured preparation failure")
    with pytest.raises(RuntimeError, match="preparation"):
        driver.execute_policy(prepare=prepare, make_native=lambda _: pytest.fail("owner constructed"),
            runtime=None, p=None, weight=0., check_static=lambda: None, attempt=attempt, clock=clock)
    assert attempt["preparation_counts_unknown"] is True


def test_interrupted_gradient_retains_unknown_prefix_without_new_native_query(tmp_path):
    # Inline setup retains the manufactured owner even when the API raises.
    p, _ = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    p = copy.deepcopy(p); p["physics"]["shape"] = [1, 1]
    deps = adapter.load_frozen_dependencies(); deps.engine.minimize = scripted((.23,))
    clock, attempt, owners = Clock(), {}, []
    def factory(problem):
        owner = ScalarNative(clock, interrupt=True); owners.append(owner); return owner
    with pytest.raises(KeyboardInterrupt):
        driver.execute_policy(prepare=lambda: SimpleNamespace(initial=np.array([.25])), make_native=factory,
            runtime=SimpleNamespace(adapter=adapter, dependencies=deps, np=np), p=p, weight=100.,
            check_static=lambda: None, attempt=attempt, clock=clock)
    assert attempt["current_native_work_unknown"] is True
    assert attempt["bounded_prefix"]["current_counter_snapshot_unknown"] is True
    assert owners[0].events[-1] == "gradient"
    row = {"run_index": 0}
    driver.persist_failure(tmp_path, row, attempt, np, 1024**2)
    saved = json.loads(gzip.decompress((tmp_path/row["failure_prefix_artifact"]).read_bytes()))
    counts = saved["attempt"]["composed_prefix"]["adapter_counts"]
    assert counts["native_gradient_dispatches"] == 1 and counts["native_gradient_completed"] == 0
    assert saved["attempt"]["construction_prefix"]["adapter_constructor_completed"] is True
    assert saved["current_native_work_unknown"] is True
    assert owners[0].events[-1] == "gradient"


def rows_with_quality(candidate=.08, baseline=.1):
    p, _ = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    rows = driver.schedule(p)
    for row in rows:
        value = baseline if row["weight"] == 0 else candidate
        row.update(final_study_eligible=True, initial_quality={"relative_physical_velocity_L2": .1},
            primary_quality={"relative_physical_velocity_L2": value}, common_raw_Phi_ratio=.5)
    return rows


def test_development_gate_requires_recovery_not_merely_initialization_stasis():
    assert driver.continuation_summary(rows_with_quality())["proposed_development_continuation_gate_pass"] is True
    assert driver.continuation_summary(rows_with_quality(candidate=.1, baseline=.7))["proposed_development_continuation_gate_pass"] is False


def test_one_late_or_missing_slot_prevents_complete_gate_credit():
    rows = rows_with_quality(); rows[0]["final_study_eligible"] = False
    assert driver.continuation_summary(rows)["proposed_development_continuation_gate_pass"] is False
    assert driver.continuation_summary(rows[:-1])["all48_eligible"] is False


def test_scalar_prior_has_correct_constant_scale_and_corner_degree():
    initial = [.25] * 6
    constant = [.25/1.1**2] * 6
    assert 100 * driver.scalar_prior(constant, initial, (2,3), .025, .1) == pytest.approx(.5)
    corner = initial.copy(); corner[0] /= 1.2**2
    assert driver.scalar_prior(corner, initial, (2,3), .025, .1) == pytest.approx(.2**2*(1+2*16)/12)


def test_output_reserve_refuses_write_and_counter_booleans_fail(tmp_path):
    with pytest.raises(OSError, match="reservation"):
        driver.artifact(tmp_path/"not_written", b"abc", limit=10, reserve=8)
    assert not (tmp_path/"not_written").exists()
    with pytest.raises(ValueError):
        driver.count_tree({"bad": True})
    with pytest.raises(ValueError):
        driver.count_tree({"bad": -1})


def test_boolean_outer_return_stamp_cannot_gain_eligibility():
    measurement, *_ = scalar_run(maximum_steps=0)
    with pytest.raises(ValueError):
        driver.return_gate(measurement, True, 8.)


def test_preparation_repeats_all8_and_binds_direct_initial_without_truth_policy_input():
    p, original = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    calls = []
    initial = np.array([.25])
    prepared = {"initial": initial, "truth": np.array([.2]), "truth_sha256": "synthetic_truth",
        "raw_model_sha256": "synthetic_raw", "preparation_costs": {"factorizations": 8, "forward_rhs": 48},
        "receivers": (0, 1), "blocks": tuple(SimpleNamespace(source_index=s, frequency_hz=f,
            label=f"{s}@{f}", observed=np.array([.01, .02], dtype=np.complex128)) for f in driver.FREQUENCIES for s in range(6))}
    def prepare_case(raw, protocol):
        calls.append(protocol["geometry"]["frequencies_hz"])
        assert protocol["geometry"]["blocks_per_model"] == 48
        return prepared
    helpers = SimpleNamespace(load_raw_inputs=lambda *a: [{"raw_model": np.array([[.2]])}]*8, prepare_case=prepare_case)
    runtime = SimpleNamespace(helpers=helpers, np=np, adapter=adapter)
    row, context = driver.schedule(p)[0], {}
    problem = driver.prepare_problem(Path("manufactured_no_files"), original, p, row, runtime, context, None)
    assert problem.initial is initial and not hasattr(problem, "truth")
    assert context["preparation_costs"] == {"factorizations": 8, "forward_rhs": 48}
    expected = context["common_binding"].copy(); expected["all8_observations_sha256"] = "corrupt"
    with pytest.raises(driver.StaleFreeze, match="bytes differ"):
        driver.prepare_problem(Path("manufactured_no_files"), original, p, row, runtime, {}, expected)
    assert calls == [driver.FREQUENCIES, driver.FREQUENCIES]


@pytest.mark.parametrize("corrupted", [None, "final_R", "final_F", "initial_model"])
def test_new_postreturn_audit_checks_R_and_F_not_rawPhi_monotonicity(corrupted):
    measurement, _owner, _attempt, _clock, runtime, p = scalar_run(points=(.21, .23))
    class Checks:
        def __init__(self):
            self.count = self.numeric_count = 0; self.failures = []
        def require(self, condition, label, *detail):
            self.count += 1
            if not condition: self.failures.append(label)
        def near(self, a, b, *, rtol, atol, context, family):
            self.numeric_count += 1
            self.require(math.isclose(a, b, rel_tol=rtol, abs_tol=atol), context)
    class Graph:
        def __init__(self, raw, protocol, checks, index):
            self.checks = checks; self.observations = {.25: np.ones((1,1))}
            self.initial = np.array([.25])
            self.factorizations = self.forward_rhs = 0; self.maximum_relative_linear_residual = 0.
        def evaluate(self, model):
            return {"objective": .5*float((model[0]-.1)**2), "model_sha256": adapter.model_sha(model)}
        def objective_check(self, actual, recorded, context):
            self.checks.near(actual, recorded, rtol=2e-9, atol=1e-12, context=context, family="rawPhi")
    runtime.audit = SimpleNamespace(Checks=Checks, IndependentCase=Graph, scalar_norm=lambda a: float(np.linalg.norm(a)),
        TOLERANCES={"initial_model_rtol": 2e-12, "initial_model_atol": 2e-14})
    _, original = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    context = {"raw_case": {"raw_model": np.array([[.2]])}, "prepared": {
        "blocks": (SimpleNamespace(frequency_hz=.25, observed=np.ones(1)),)}}
    if corrupted == "initial_model":
        measurement["models"]["initial_model"] = np.array([.26])
        header = measurement["payload"]["initial_native_header"]["native_data_receipt_header"]
        header.update(objective=.5*(.26-.1)**2, model_sha256=adapter.model_sha(np.array([.26])))
        own_R = driver.scalar_prior(measurement["models"]["final_model"], np.array([.26]), (1,1), .025, .1)
        measurement["payload"]["final_R"] = own_R
        measurement["payload"]["final_F"] = measurement["payload"]["final_raw_native_header"]["objective"] / header["objective"] + 100*own_R
    elif corrupted:
        measurement["payload"][corrupted] += 1.
    audit = driver.offline_check(context, measurement, p, original, runtime, 0)
    assert bool(audit["failures"]) is bool(corrupted)
    assert "fresh all8 observations" not in audit["failures"]
    if corrupted == "initial_model":
        assert audit["failures"] == ["initial vector versus independent registered initializer"]
    elif corrupted:
        assert ("new scalar prior R" if corrupted == "final_R" else "new composed F") in audit["failures"]


def test_duplicate_slot_cannot_satisfy_full48_gate():
    rows = rows_with_quality()
    rows[1] = copy.deepcopy(rows[0])
    assert driver.continuation_summary(rows)["all48_eligible"] is False


def test_failure_artifact_retains_all_interrupt_construction_and_metric_branches(tmp_path):
    attempt = {"global_started": 0., "phase": "interrupted_optimizer", "current_native_work_unknown": True,
        "bounded_prefix": {"initial_model": np.array([.25]), "current_counter_snapshot_unknown": True},
        "composed_prefix": {"adapter_counts": {"native_gradient_dispatches": 1, "native_gradient_completed": 0}},
        "construction_prefix": {"native_factory_attempted": True, "adapter_constructor_completed": True},
        "metric_tail_counts": {"native_evaluate_attempts": 1, "native_evaluate_completed": 0}}
    row = {"run_index": 0}
    driver.persist_failure(tmp_path, row, attempt, np, 1024**2)
    saved = json.loads(gzip.decompress((tmp_path/row["failure_prefix_artifact"]).read_bytes()))
    assert set(saved["attempt"]) == set(attempt)
    assert saved["attempt"]["composed_prefix"]["adapter_counts"]["native_gradient_completed"] == 0
    assert saved["attempt"]["construction_prefix"]["native_factory_attempted"] is True
    assert saved["attempt"]["metric_tail_counts"]["native_evaluate_attempts"] == 1
    assert saved["current_native_work_unknown"] is True and saved["completed_policy_return_observed"] is False


def test_serialization_failure_persists_completed_final_native_snapshot_and_tail(tmp_path, monkeypatch):
    original, attempt = driver.encoded, {}
    def fail_return_only(value):
        if "engine" in value:
            raise TypeError("manufactured return serialization failure")
        return original(value)
    monkeypatch.setattr(driver, "encoded", fail_return_only)
    with pytest.raises(TypeError, match="serialization"):
        scalar_run(maximum_steps=0, attempt_out=attempt)
    snapshot = copy.deepcopy(attempt["final_native_snapshot"])
    row = {"run_index": 0}
    driver.persist_failure(tmp_path, row, attempt, np, 1024**2)
    saved = json.loads(gzip.decompress((tmp_path/row["failure_prefix_artifact"]).read_bytes()))
    assert saved["attempt"]["final_native_snapshot"] == snapshot
    assert saved["attempt"]["metric_tail_counts"]["native_evaluate_completed"] == 1
    assert saved["attempt"]["metric_tail_counts"]["prior_evaluations_completed"] == 1
    assert saved["attempt"]["return_payload_before_serialization"]["final_model"] == [.25]
    assert saved["attempt"]["final_metrics"]["final_F"] == 1.
    assert saved["final_native_snapshot_available"] is True
    assert saved["completed_policy_return_observed"] is False


def test_construction_interrupt_prefix_survives_actual_failure_persistence(tmp_path):
    p, _ = driver.load_protocol(Path(driver.__file__).with_name("protocol.json"))
    p = copy.deepcopy(p); p["physics"]["shape"] = [1,1]
    deps = adapter.load_frozen_dependencies()
    runtime, attempt, clock = SimpleNamespace(adapter=adapter, dependencies=deps, np=np), {}, Clock()
    def interrupted_factory(problem):
        raise KeyboardInterrupt("manufactured construction interrupt")
    with pytest.raises(KeyboardInterrupt):
        driver.execute_policy(prepare=lambda: SimpleNamespace(initial=np.array([.25])), make_native=interrupted_factory,
            runtime=runtime, p=p, weight=100., check_static=lambda: None, attempt=attempt, clock=clock)
    row = {"run_index": 0}
    driver.persist_failure(tmp_path, row, attempt, np, 1024**2)
    saved = json.loads(gzip.decompress((tmp_path/row["failure_prefix_artifact"]).read_bytes()))
    assert saved["attempt"]["construction_prefix"]["native_factory_attempted"] is True
    assert saved["attempt"]["construction_prefix"]["native_factory_returned"] is False
    assert saved["attempt"]["bounded_prefix"]["current_counter_snapshot_unknown"] is True
