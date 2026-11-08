"""Abstract clock and synthetic one-variable checks; no project arrays/timings."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from wavecert.optimization.paid_descent import TrajectoryOptions, run_exact_lbfgsb, run_paid_descent

import run_engineering_benchmark as driver


class ManualClock:
    def __init__(self, value=100.):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def candidate():
    path = Path(__file__).resolve().parents[2]/"wavecert-fwi/results/provisional_software_benchmark/matched_cost_pilot/protocol.json"
    return json.loads(path.read_text())


class ToyOracle:
    def __init__(self, clock):
        self.clock, self.merits, self.gradients = clock, 0, 0
        clock.advance(.1)

    def evaluate(self, model):
        self.clock.advance(1.)
        self.merits += 1
        return SimpleNamespace(model=np.asarray(model).copy(), objective=float(np.sum((model-.2)**2)))

    def gradient(self, receipt):
        self.clock.advance(.2)
        self.gradients += 1
        return SimpleNamespace(gradient=2*(receipt.model-.2))

    def accept(self, receipt):
        pass

    def cost_snapshot(self):
        return dict(merits=self.merits, gradients=self.gradients)


class ToyHelpers:
    def __init__(self, clock, input_seconds=1.25):
        self.clock, self.input_seconds = clock, input_seconds
        self.loaded, self.prepared = 0, 0

    def load_raw_inputs(self, root, protocol):
        self.clock.advance(self.input_seconds)
        self.loaded += 1
        return [dict(marker="synthetic fixture")]

    def prepare_case(self, case, protocol):
        self.clock.advance(.75)
        self.prepared += 1
        return dict(initial=np.array([.8]), truth=np.array([.2]), blocks=("observations_only",), receivers=(0,),
            initial_sha256="synthetic_initial", preparation_costs=dict(factorizations=4, forward_rhs=24))


class ToyAdapter:
    def __init__(self, clock, tail=.4):
        self.clock, self.tail, self.oracles = clock, tail, []

    def run_control(self, method, **kwargs):
        assert "truth" not in kwargs and "prepared" not in kwargs

        def factory():
            oracle = ToyOracle(self.clock)
            self.oracles.append(oracle)
            return oracle

        common = dict(oracle_factory=factory, m0=kwargs["model"], lower_bound=kwargs["lower_bound"],
            upper_bound=kwargs["upper_bound"], options=kwargs["options"], clock=kwargs["clock"])
        result = (run_exact_lbfgsb(**common) if method == "exact_lu_lbfgsb"
                  else run_paid_descent(**common, proxy_factory=None))
        # Simulated finishing work AFTER the engine elapsed timestamp, before API return.
        self.clock.advance(self.tail)
        return result


def timed_fixture(*, input_seconds=1.25, method="exact_lu_projected"):
    clock = ManualClock()
    helpers, adapter, context = ToyHelpers(clock, input_seconds), ToyAdapter(clock), {}
    row = dict(method=method, target_ratio=None, case_index=0)
    phases = []
    timed = driver.execute_timed(Path("synthetic"), candidate(), row, helpers, adapter, TrajectoryOptions,
        check_static=lambda: None, context=context, phase=lambda name, details: phases.append((name, details)), clock=clock)
    return timed, helpers, adapter, context, phases


def test_anchored_clock_uses_true_global_start_once_then_actual_current_time():
    clock = ManualClock(10.)
    anchored = driver.AnchoredClock(2., clock)
    assert anchored() == 2.
    assert anchored() == 10.
    clock.advance(3.)
    assert anchored() == 13. and anchored.calls == 3


def test_real_preparation_prefix_is_in_engine_timestamps_without_double_charge():
    timed, helpers, adapter, _, _ = timed_fixture()
    result, timing = timed["result"], timed["timing"]
    assert helpers.loaded == helpers.prepared == 1
    assert timing["input_loading_verification_seconds"] == pytest.approx(1.25)
    assert timing["refinement_initialization_observation_seconds"] == pytest.approx(.75)
    assert timing["pre_control_elapsed_seconds"] == pytest.approx(2.)
    assert result["setup_seconds"] == pytest.approx(2.1)
    assert result["checkpoints"][0]["elapsed_seconds"] == pytest.approx(3.1)
    assert all(row["elapsed_seconds"] <= 8. for row in result["checkpoints"])
    assert not any(row["elapsed_seconds"] <= 2. for row in result["checkpoints"])
    assert timing["engine_finish_tail_seconds"] == pytest.approx(.4)
    assert timing["actual_start_to_control_return_seconds"] == pytest.approx(result["total_online_seconds"]+.4)
    assert timing["preparation_physics_costs"] == dict(factorizations=4, forward_rhs=24)
    assert adapter.oracles[0].merits > 0


@pytest.mark.parametrize("method", ["exact_lu_projected", "exact_lu_lbfgsb"])
def test_preparation_exceeding_budget_has_no_paid_checkpoint_or_merit_credit(method):
    timed, _, adapter, _, _ = timed_fixture(input_seconds=8.25, method=method)
    result = timed["result"]
    assert result["status"] == "budget_exhausted" and not result["checkpoints"]
    assert result["initial_objective"] is None
    assert result["setup_seconds"] == pytest.approx(9.1)
    assert adapter.oracles[0].merits == adapter.oracles[0].gradients == 0
    assert not timed["timing"]["fully_returned_within_max_budget"]


def returned_result(status="objective_target", initial=10., final=4.):
    return dict(status=status, costs_complete=True, initial_objective=initial, final_objective=final,
        checkpoints=[dict(within_budget=True, objective=4., elapsed_seconds=7.7)])


def test_target_must_return_by_budget_even_when_internal_target_was_on_time():
    row = dict(target_ratio=.5)
    late = driver.end_to_end_target(returned_result(), row, 8.2, 8.)
    assert late["reached_by_returned_objective"] and not late["eligible"]
    assert late["elapsed_seconds"] is None and late["reason"] == "target_returned_after_budget"
    assert late["internal_first_target_checkpoint_seconds"] == 7.7
    timely = driver.end_to_end_target(returned_result(), row, 7.9, 8.)
    assert timely["eligible"] and timely["elapsed_seconds"] == 7.9


def test_failed_iteration_limited_and_trivial_initial_targets_are_not_speedup_events():
    row = dict(target_ratio=.5)
    assert not driver.end_to_end_target(returned_result(status="failed"), row, 2., 8.)["eligible"]
    limited = driver.end_to_end_target(returned_result(status="maximum_steps", final=8.), row, 2., 8.)
    assert limited["reason"] == "unreached_maximum_steps" and limited["observation_end_seconds"] == 2.
    trivial = driver.end_to_end_target(returned_result(initial=0., final=0.), row, 2., 8.)
    assert "trivial" in trivial["reason"] and not trivial["eligible"]


def test_offline_quality_reports_both_parameters_and_gates_unpaid_or_late_states():
    timed = dict(prepared=dict(truth=np.array([.2])), result=dict(final_model=np.array([.8])),
        end_to_end_target=dict(eligible=False))
    endpoints = [dict(budget_seconds=1., completed_paid_checkpoint=False, model=np.array([.8])),
                 dict(budget_seconds=2., completed_paid_checkpoint=True, model=np.array([.8]))]
    quality = driver.offline_reconstruction_quality(timed, endpoints)
    assert quality["internal_budget_states"][0]["quality"] is None
    paid = quality["internal_budget_states"][1]["quality"]
    assert paid["relative_squared_slowness_L2"] == pytest.approx(3.)
    assert paid["relative_physical_velocity_L2"] == pytest.approx(.5)
    assert quality["fully_returned_target_state"] is None
    timed["end_to_end_target"]["eligible"] = True
    assert driver.offline_reconstruction_quality(timed, endpoints)["fully_returned_target_state"] == paid


def test_all360_runs_preserve_original_order_and_only_add_two_fixed_stopping_ratios():
    protocol = candidate()
    original = [dict(run_index=i, repetition=i//40, case_index=(i//5) % 8,
        order_position=i % 5, method=protocol["methods"][i % 5]) for i in range(120)]
    original_copy = copy.deepcopy(original)
    schedule = driver.make_schedule(protocol, original)
    assert original == original_copy and len(schedule) == 360
    for block, ratio in enumerate([None, .5, .25]):
        for i, source in enumerate(original):
            row = schedule[block*120+i]
            assert row["original_schedule_index"] == i and row["method"] == source["method"]
            assert row["target_ratio"] == ratio and row["status"] == "unattempted"
    assert protocol["project_execution_authorized"] is False
    assert protocol["trajectory"]["relative_objective_target"] is None


def test_default_cli_does_not_load_candidate_inputs_or_execute(monkeypatch, capsys):
    monkeypatch.setattr(driver, "load_candidate", lambda *_: pytest.fail("candidate execution reached"))
    assert driver.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "metadata_only_no_project_execution"


def test_failed_preparation_keeps_unknown_counters_and_never_constructs_policy():
    clock, context = ManualClock(), {}
    helpers, adapter = ToyHelpers(clock), ToyAdapter(clock)

    def failure(*_):
        clock.advance(.5)
        raise OSError("synthetic missing inputs")

    helpers.load_raw_inputs = failure
    with pytest.raises(OSError, match="synthetic missing"):
        driver.execute_timed(Path("synthetic"), candidate(), dict(method="exact_lu_projected", target_ratio=None, case_index=0),
            helpers, adapter, TrajectoryOptions, check_static=lambda: None, context=context,
            phase=lambda *_: None, clock=clock)
    assert context["phase"] == "input_loading_and_verification"
    assert context["preparation_physics_costs"] is None and not adapter.oracles


def test_timed_phase_omits_360_history_and_interrupted_progress_is_recoverable(monkeypatch, tmp_path):
    protocol = candidate()
    source_schedule = [dict(run_index=i, repetition=i//40, case_index=(i//5) % 8,
        order_position=i % 5, method=protocol["methods"][i % 5]) for i in range(120)]
    helpers = SimpleNamespace(scheduled_runs=lambda _: source_schedule)
    monkeypatch.setattr(driver, "load_candidate", lambda *_: (helpers, None, TrajectoryOptions, protocol, b"synthetic metadata"))
    monkeypatch.setattr(driver, "static_receipt", lambda *_: {"scope": "synthetic source freeze"})
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda *_: SimpleNamespace(free=1024**3))
    for name, value in protocol["threads"].items():
        monkeypatch.setenv(name, value)
    output = tmp_path/"interrupted"

    def interrupt(*args, context, phase, **kwargs):
        plans = json.loads((output/"schedule.json").read_text())
        assert len(plans) == 360 and all(r["status"] == "unattempted" for r in plans)
        before_phase = (output/"status.json").read_bytes()
        context.update(phase="policy_call", preparation_physics_costs=None, inside_timed_interval=True)
        phase("policy_call", {})
        assert (output/"status.json").read_bytes() == before_phase
        status = json.loads((output/"status.json").read_text())
        assert len(status["runs"]) == 360 and status["runs"][0]["status"] == "attempted"
        assert status["runs"][0]["phase"] == "before_preparation"
        progress_path = output/status["runs"][0]["progress_artifact"]
        progress_raw = progress_path.read_bytes()
        progress = json.loads(progress_raw)
        assert progress["phase"] == "policy_call" and progress["run_index"] == 0
        assert "runs" not in progress and "identity" not in progress and len(progress_raw) < 8192
        assert context["timed_progress_write_seconds"] >= 0
        assert all(r["status"] == "unattempted" for r in status["runs"][1:])
        raise KeyboardInterrupt("synthetic interruption; no project work")

    monkeypatch.setattr(driver, "execute_timed", interrupt)
    addendum = Path(driver.__file__).with_name("execution_addendum.json")
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        driver.run(tmp_path, output, addendum, driver.sha(addendum.read_bytes()), driver.sha(Path(driver.__file__).read_bytes()))
    status = json.loads((output/"status.json").read_text())
    assert status["status"] == "failed_or_interrupted"
    assert status["runs"][0]["status"] == "interrupted" and status["runs"][0]["phase_at_failure"] == "policy_call"
    assert all(r["status"] == "unattempted" for r in status["runs"][1:])
    progress = json.loads((output/status["runs"][0]["progress_artifact"]).read_text())
    assert progress["status"] == "interrupted" and progress["phase_at_failure"] == "policy_call"
    assert not (output/"complete_receipt.json").exists()
