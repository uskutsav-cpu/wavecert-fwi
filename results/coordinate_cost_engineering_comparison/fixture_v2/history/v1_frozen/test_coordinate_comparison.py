"""Pure abstract-clock/coordinate/retention mocks; no project arrays or solves."""
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_local(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = load_local("coordinate_driver_mock_test_unique", "run_coordinate_comparison.py")
coordinates = load_local("coordinate_map_mock_test_unique", "velocity_oracle.py")


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, value):
        self.now += value


def protocol_values():
    return driver.load_protocol(Path(__file__).with_name("protocol.json"))


def fake_runtime(clock, *, load_seconds=.2, finish_seconds=.2, fail_conversion=False, engine_failed=False):
    seen = {}
    prepared = dict(initial=np.array([.17, .2, .24]), truth=np.array([.18, .21, .25]),
        blocks=[SimpleNamespace(source_index=1, frequency_hz=2., observed=np.array([1 + 2j]))],
        receivers=[1], preparation_costs=dict(factorizations=4, forward_rhs=24),
        truth_sha256="synthetic_truth", raw_model_sha256="synthetic_raw")
    def load(root, original):
        clock.advance(load_seconds)
        return [dict(raw_model=np.ones((2, 2)), identity={}) for _ in range(8)]
    def prepare(case, original):
        clock.advance(.3)
        return prepared
    class Native:
        def __init__(self, *args):
            clock.advance(.05)
            self.outputs = 0
            self.contract = dict(kind="synthetic_native")
        def cost_snapshot(self):
            clock.advance(.1)
            return dict(physics=dict(factorizations=1), oracle=dict(evaluations=1))
    class Velocity:
        def __init__(self, native, *, velocity_units):
            self.native, self.conversions, self.conversion_attempts = native, 0, 0
            self.initial_model_record = None
            self.contract = dict(kind="synthetic_velocity", velocity_units=velocity_units)
        def to_native_model(self, values):
            self.conversion_attempts += 1
            if fail_conversion:
                clock.advance(.4)
                raise ValueError("synthetic final conversion failure")
            clock.advance(.4)
            self.conversions += 1
            model = coordinates.velocity_to_slowness(values)
            return model, dict(actual_native_model_sha256=coordinates.model_sha256(model))
        def cost_snapshot(self):
            result = self.native.cost_snapshot()
            result["velocity_coordinates"] = dict(endpoint_conversions=self.conversions, conversion_attempts=self.conversion_attempts)
            return result
    def engine(*, oracle_factory, m0, options, clock, **kwargs):
        start = clock()
        seen["anchored_first"] = start
        policy = oracle_factory()
        seen.update(initial=np.array(m0, copy=True), budget=options.budget_seconds)
        if clock() - start > options.budget_seconds:
            costs = dict(oracle=policy.cost_snapshot(), proposal={})
            elapsed = clock() - start
            return dict(status="budget_exhausted", costs_complete=True, final_model=np.array(m0, copy=True),
                initial_objective=None, final_objective=None, checkpoints=[], total_online_costs=costs,
                total_online_seconds=elapsed, budget_overshoot_seconds=max(0., elapsed - options.budget_seconds))
        clock_owner.advance(.8)
        if isinstance(policy, Velocity):
            policy.initial_model_record = dict(velocity_model_sha256=coordinates.model_sha256(m0),
                actual_native_model_sha256=coordinates.model_sha256(coordinates.velocity_to_slowness(m0)))
        costs = dict(oracle=policy.cost_snapshot(), proposal={})
        elapsed = clock() - start
        clock_owner.advance(finish_seconds)
        final = np.array(m0, copy=True) * 1.001
        return dict(status="failed" if engine_failed else "solver_terminated", costs_complete=True,
            final_model=final, initial_objective=4., final_objective=2.,
            checkpoints=[dict(model=np.array(m0, copy=True), model_sha256=coordinates.model_sha256(m0), objective=4.),
                         dict(model=final, model_sha256=coordinates.model_sha256(final), objective=2.)],
            total_online_costs=costs, total_online_seconds=elapsed, budget_overshoot_seconds=max(0., elapsed - options.budget_seconds))
    clock_owner = clock
    vp = SimpleNamespace(**{name: getattr(coordinates, name) for name in
                           ["start_from_slowness", "velocity_to_slowness", "slowness_bounds_to_velocity", "model_sha256"]},
                         VelocityMeritOracle=Velocity)
    def compact(result):
        return dict(status=result["status"], final_parameter_sha256=coordinates.model_sha256(result["final_model"]))
    runtime = SimpleNamespace(helpers=SimpleNamespace(load_raw_inputs=load, prepare_case=prepare, compact_trajectory=compact),
        physics=SimpleNamespace(Helmholtz2D=lambda **kwargs: None), native=SimpleNamespace(ExactMeritOracle=Native),
        engine=SimpleNamespace(run_exact_lbfgsb=engine, TrajectoryOptions=lambda **kwargs: SimpleNamespace(**kwargs)),
        vp=vp, np=np)
    return runtime, seen, prepared


def execute_fake(method, **kwargs):
    protocol, _, original, _ = protocol_values()
    clock = Clock()
    runtime, seen, prepared = fake_runtime(clock, **kwargs)
    context, phases = {}, []
    row = next(row for row in driver.make_schedule(protocol) if row["method"] == method)
    def check_static():
        clock.advance(.1)
    driver.execute_timed(Path("/synthetic"), protocol, original, row, runtime, context,
        lambda name, detail: phases.append((name, detail)), check_static, clock=clock)
    returned = clock()
    context["inside_timed_interval"] = False
    return context, driver.timing_record(context, returned, protocol), runtime, seen, prepared, phases


def test_schedule_exact_48_unique_and_seeded_without_input_reads():
    protocol, _, _, _ = protocol_values()
    rows = driver.make_schedule(protocol)
    assert rows == driver.make_schedule(protocol) and len(rows) == 48
    assert len({(r["case_index"], r["method"], r["repetition"]) for r in rows}) == 48
    assert all(r["status"] == "unattempted" for r in rows)
    for index in range(0, 48, 2):
        assert {rows[index]["method"], rows[index + 1]["method"]} == set(protocol["methods"])


def test_true_prefix_anchor_and_final_conversion_paid_without_double_count():
    context, timing, _, seen, _, _ = execute_fake("physical_v_lbfgsb")
    assert seen["anchored_first"] == 0 and seen["budget"] == 7.5
    assert timing["pre_engine_seconds"] == pytest.approx(.6)
    assert timing["engine_reported_seconds_including_preparation"] == pytest.approx(1.55)
    assert timing["engine_finish_tail_seconds"] == pytest.approx(.2)
    assert timing["actual_api_return_seconds"] == pytest.approx(2.25)
    assert timing["api_output_tail_seconds"] == pytest.approx(.5)
    assert context["api_cost_snapshot"]["oracle"]["velocity_coordinates"]["endpoint_conversions"] == 1
    assert context["result"]["total_online_costs"]["oracle"]["velocity_coordinates"]["endpoint_conversions"] == 0
    assert timing["runtime_gate"]["runtime_eligible"]


def test_both_coordinate_arms_start_same_actual_native_m_and_keep_output_tags():
    m, _, _, seen_m, prepared, _ = execute_fake("native_m_lbfgsb")
    v, _, _, seen_v, _, _ = execute_fake("physical_v_lbfgsb")
    velocity, common, record = coordinates.start_from_slowness(prepared["initial"])
    assert np.array_equal(seen_m["initial"], common)
    assert np.array_equal(seen_v["initial"], velocity)
    assert np.array_equal(coordinates.velocity_to_slowness(seen_v["initial"]), common)
    assert m["common_binding"] == v["common_binding"]
    assert record["requested_native_model_sha256"] != record["actual_native_model_sha256"]
    assert m["output_provenance"]["optimizer_coordinates"] == "squared_slowness"
    assert v["output_provenance"]["optimizer_coordinates"] == "physical_velocity"
    assert not np.array_equal(v["final_parameter"], v["final_native_m"])


def test_prefix_over_budget_has_no_unpaid_final_quality_credit():
    context, timing, _, seen, _, _ = execute_fake("native_m_lbfgsb", load_seconds=8.)
    assert seen["budget"] == 7.5
    assert context["result"]["checkpoints"] == []
    assert not timing["runtime_gate"]["runtime_eligible"]
    assert timing["runtime_gate"]["observed_event_seconds"] is None
    assert timing["runtime_gate"]["reason"] == "no_paid_final_state"


def test_slow_finish_or_engine_failure_cannot_get_return_credit():
    _, late, _, seen, _, _ = execute_fake("physical_v_lbfgsb", finish_seconds=8.)
    assert late["engine_reported_seconds_including_preparation"] < 7.5
    assert late["actual_api_return_seconds"] > 8 and seen["budget"] == 7.5
    assert not late["runtime_gate"]["runtime_eligible"] and late["runtime_gate"]["observed_event_seconds"] is None
    _, failed, _, _, _, _ = execute_fake("native_m_lbfgsb", engine_failed=True)
    assert not failed["runtime_gate"]["runtime_eligible"]
    assert failed["runtime_gate"]["reason"] == "failed_or_incomplete_method"


def test_failed_final_conversion_preserves_returned_parameter_and_costs(tmp_path):
    protocol, _, original, _ = protocol_values()
    clock = Clock()
    runtime, _, _ = fake_runtime(clock, fail_conversion=True)
    row = next(r for r in driver.make_schedule(protocol) if r["method"] == "physical_v_lbfgsb")
    context = {}
    with pytest.raises(ValueError, match="conversion failure"):
        driver.execute_timed(tmp_path, protocol, original, row, runtime, context, lambda *args: None, lambda: None, clock=clock)
    assert "result" in context and "final_native_m" not in context
    driver.preserve_failure(tmp_path, row, context, runtime)
    record = json.loads(gzip.decompress((tmp_path / row["failure_artifact"]).read_bytes()))
    assert not record["in_flight_engine_counters_unknown"]
    assert not record["latest_policy_counter_snapshot_unknown"]
    assert record["latest_observed_policy_cost_snapshot"]["velocity_coordinates"]["conversion_attempts"] == 1
    assert record["earlier_engine_cost_snapshot"]["oracle"]["velocity_coordinates"]["conversion_attempts"] == 0
    assert not record["observed_counters_complete_for_successful_api"]
    with np.load(tmp_path / record["failure_models_artifact"], allow_pickle=False) as saved:
        assert np.array_equal(saved["returned_parameter"], context["result"]["final_model"])


def test_timed_progress_never_serializes_growing_history(tmp_path):
    row = dict(run_index=0, case_index=0, method="physical_v_lbfgsb", phase="policy_call",
               progress_artifact="run_000_progress.json", history=[dict(run=i) for i in range(1000)])
    context = dict(inside_timed_interval=True)
    driver.progress(tmp_path, row, context, clock=lambda: 0.)
    value = json.loads((tmp_path / row["progress_artifact"]).read_bytes())
    assert "history" not in value and len((tmp_path / row["progress_artifact"]).read_bytes()) < 2000
    assert not (tmp_path / "status.json").exists()


def test_changed_or_clipped_first_paid_state_fails_common_binding():
    context, _, runtime, _, _, _ = execute_fake("physical_v_lbfgsb")
    result = context["result"]
    first = result["checkpoints"][0]
    first["model"] = np.array(first["model"], copy=True) * 1.001
    first["model_sha256"] = coordinates.model_sha256(first["model"])
    validation = driver.validate_paid_initial(result, context["initial_parameter"], context["common_binding"],
                                             context["policy"], True, runtime)
    assert validation["paid_checkpoint_available"] and not validation["passes"]


def test_all48_attempt_prefix_and_interrupted_phase_retained(monkeypatch, tmp_path):
    protocol, raw, _, _ = protocol_values()
    clock = Clock()
    runtime, _, _ = fake_runtime(clock)
    monkeypatch.setattr(driver, "load_runtime", lambda *args: runtime)
    monkeypatch.setattr(driver, "static_receipt", lambda *args: dict(frozen="synthetic"))
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda *args: SimpleNamespace(free=2 ** 30))
    for key, value in protocol["threads"].items():
        monkeypatch.setenv(key, value)
    output, root = tmp_path / "output", tmp_path / "repository"
    root.mkdir()
    def interrupted(root, protocol, original, row, runtime, context, phase, check_static, **kwargs):
        before = json.loads((output / "status.json").read_bytes())
        assert len(before["runs"]) == 48 and before["runs"][0]["status"] == "attempted"
        assert all(r["status"] == "unattempted" for r in before["runs"][1:])
        assert row["progress_artifact"] in {p.name for p in output.iterdir()}
        context.update(global_started=driver.perf_counter(), inside_timed_interval=True)
        phase("synthetic_interrupted_policy", {})
        raise KeyboardInterrupt("synthetic prefix interruption")
    monkeypatch.setattr(driver, "execute_timed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        driver.run(root, output, Path(__file__).with_name("protocol.json"), driver.sha(raw), driver.sha(Path(driver.__file__).read_bytes()))
    after = json.loads((output / "status.json").read_bytes())
    assert after["status"] == "failed_or_interrupted" and len(after["runs"]) == 48
    assert after["runs"][0]["status"] == "interrupted" and after["runs"][0]["in_flight_engine_counters_unknown"]
    assert all(r["status"] == "unattempted" for r in after["runs"][1:])
    phase = json.loads((output / after["runs"][0]["progress_artifact"]).read_bytes())
    assert phase["phase"] == "synthetic_interrupted_policy" and phase["preparation_physics_costs"] is None
    assert not (output / "complete_receipt.json").exists()


def test_default_cli_is_metadata_only(monkeypatch, capsys):
    monkeypatch.setattr(driver, "load_runtime", lambda *args: pytest.fail("must not import execution runtime"))
    assert driver.main([]) == 0
    assert "metadata_only_no_project_arrays" in capsys.readouterr().out
