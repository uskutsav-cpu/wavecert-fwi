"""Manufactured equivalence controls, never project data or runtime evidence.

Scripted solvers replace bindings only in the staged engine and a private
validation copy of the frozen old source. Production modules stay untouched.
One tiny real Helmholtz control uses the actual SciPy optimizer and exact oracle.
"""
from __future__ import annotations

import gzip
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
        self.reads = []

    def __call__(self):
        self.reads.append(self.value)
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
            solve.callback_invocations += 1
            callback(model)
        count = min(len(candidates), options["maxiter"])
        return SimpleNamespace(success=True, status=0, message="manufactured native return",
                               nit=count, nfev=count, njev=count if gradient else 0)
    solve.callback_invocations = 0
    return solve


def reference_samples(records):
    return records[:8]+records[max(8, len(records)-8):]


def reference_summary(records):
    """Independent implementation of the published length-delimited chain."""
    state = hashlib.sha256(b"bounded-exact-record-stream-v1\0").digest()
    for index, record in enumerate(records, 1):
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        state = hashlib.sha256(state+index.to_bytes(8, "big")+len(payload).to_bytes(8, "big")+payload).digest()
    retained = len(reference_samples(records))
    return dict(total_count=len(records), chain_sha256=state.hex(), retained_count=retained,
        omitted_count=len(records)-retained, head_limit=8, tail_limit=8,
        finalized_count=len(records), pending_record=None, pending_record_unsealed=False)


def normalized(result):
    """Compare samples/counts/chains with independent full old-engine records."""
    extra = {"initial_model", "checkpoint_vectors_retained", "native_callback_count",
             "maximum_steps_safety_cap", "scipy_maxfun_default", "scipy_maxfun_passed_explicitly",
             "last_native_callback_attempt", "trace_retention", "native_callback_invocations",
             "native_callback_point_records", "native_callback_count_semantics"}
    value = {k: v for k, v in result.items() if k not in extra}
    if "checkpoint_trace" not in result:
        checkpoints = [{k: v for k, v in checkpoint.items() if k != "model"}
                       for checkpoint in result["checkpoints"]]
        steps, library_records = [], []
        for original in result["steps"]:
            step = dict(original)
            library = step.get("library_evaluations", [])
            library_records.extend(library)
            step["library_evaluations"] = reference_samples(library)
            step["library_evaluations_summary"] = reference_summary(library)
            steps.append(step)
        value.update(checkpoints=reference_samples(checkpoints), checkpoint_trace=reference_summary(checkpoints),
            steps=reference_samples(steps), step_trace=reference_summary(steps),
            all_library_evaluation_trace={key: reference_summary(library_records)[key]
                                          for key in ["total_count", "chain_sha256"]})
    return bounded._strict_metadata(value)


def paired_toy(candidates, *, initial=(.8,), target=(.3,), opts=None, gradient=True, **oracle_options):
    old = load_old()
    results, owners, clocks, callback_counts = [], [], [], []
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
            clocks.append(clock)
            callback_counts.append(module.minimize.callback_invocations)
    finally:
        bounded.minimize = original_bounded_solver
    assert owners[0].events == owners[1].events
    assert owners[0].counts == owners[1].counts
    assert clocks[0].reads == clocks[1].reads
    assert callback_counts[0] == callback_counts[1] == results[1]["native_callback_count"]
    assert results[1]["native_callback_invocations"] == callback_counts[0]
    assert results[1]["native_callback_point_records"] == sum("native_callback" in step for step in results[0]["steps"])
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
    assert len(new["checkpoints"]) == 16 and new["native_callback_count"] == 128
    assert new["checkpoint_trace"]["total_count"] == 129
    assert new["checkpoint_trace"]["omitted_count"] == 113
    assert len(new["steps"]) == 16 and new["step_trace"]["total_count"] == 128
    assert new["all_library_evaluation_trace"]["total_count"] == 256
    assert new["checkpoints"][0]["kind"] == "initial"
    assert new["checkpoints"][-1]["model_sha256"] == bounded._digest(new["final_model"])
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
    old, requested, clock_reads = load_old(), [], []
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
            clock_reads.append(clock.reads)
            if module is bounded:
                prefix = caught.value.bounded_exact_pending_prefix
                assert prefix["status"] == "unfinalized_engine_prefix" and not prefix["costs_complete"]
                assert prefix["current_counter_snapshot_unknown"] and prefix["no_completed_return_claim"]
                assert np.array_equal(prefix["final_model"], [.8])
                assert prefix["checkpoints"][0]["kind"] == "initial"
                library = prefix["active_step"]["library_evaluations_trace"]
                assert library["total_count"] == 1 and library["pending_record_unsealed"]
                assert library["pending_record"]["completed"] is True
    finally:
        bounded.minimize = original_bounded_solver
    assert requested[0] == requested[1]
    assert clock_reads[0] == clock_reads[1]


def test_mutable_library_stream_seals_final_records_and_bounds_pending_slot():
    stream, expected = bounded._Stream(), []
    for index in range(40):
        event = dict(index=index, completed=False)
        stream.append(event)
        assert stream.summary()["pending_record_unsealed"]
        assert len(stream.head)+len(stream.tail)+1 <= 17
        event.update(completed=True, result=index**2)
        expected.append(dict(event))
    stream.finalize()
    assert stream.records() == reference_samples(expected)
    assert stream.summary() == reference_summary(expected)
    assert stream.summary()["total_count"] == 40 and stream.summary()["omitted_count"] == 24


def test_callback_aborted_by_entry_guard_still_counts_as_native_callback():
    old, results, reads, events = load_old(), [], [], []
    original_bounded_solver = bounded.minimize
    try:
        for module in (old, bounded):
            clock, made = Clock(), []
            def solver(function, initial, *, callback, clock=clock, **kwargs):
                clock.advance(2.)
                callback(np.full_like(initial, .6))
                pytest.fail("entry guard must stop a late callback")
            module.minimize = solver
            def factory(clock=clock, made=made):
                owner = Oracle(clock, [.3])
                made.append(owner)
                return owner
            result = module.run_exact_lbfgsb(oracle_factory=factory, m0=np.array([.8]), lower_bound=.1,
                upper_bound=1., options=options(budget_seconds=1.), clock=clock)
            results.append(result)
            reads.append(clock.reads)
            events.append(made[0].events)
    finally:
        bounded.minimize = original_bounded_solver
    assert normalized(results[0]) == normalized(results[1])
    assert reads[0] == reads[1] and events[0] == events[1]
    assert results[1]["status"] == "budget_exhausted" and not results[1]["steps"]
    assert results[1]["native_callback_count"] == 1
    assert results[1]["native_callback_invocations"] == 1
    assert results[1]["native_callback_point_records"] == 0
    assert results[1]["last_native_callback_attempt"]["candidate_sha256"] == bounded._digest(np.array([.6]))


def test_failed_counter_finalization_preserves_unknown_unsealed_prefix_without_queries():
    class FailedSnapshot(Oracle):
        def cost_snapshot(self):
            if self.counts["gradients"]:
                self.events.append(("unavailable_cost_snapshot",))
                raise RuntimeError("manufactured unavailable final counters")
            return super().cost_snapshot()
    old, reads, events = load_old(), [], []
    original_bounded_solver = bounded.minimize
    try:
        for module in (old, bounded):
            module.minimize = scripted([np.array([.6])])
            clock, made = Clock(), []
            def factory(clock=clock, made=made):
                owner = FailedSnapshot(clock, [.3])
                made.append(owner)
                return owner
            with pytest.raises(RuntimeError, match="manufactured unavailable") as caught:
                module.run_exact_lbfgsb(oracle_factory=factory, m0=np.array([.8]), lower_bound=.1,
                    upper_bound=1., options=options(), clock=clock)
            reads.append(clock.reads)
            events.append(made[0].events)
            if module is bounded:
                prefix = caught.value.bounded_exact_pending_prefix
                assert prefix["current_counter_snapshot_unknown"] and not prefix["costs_complete"]
                assert prefix["no_completed_return_claim"] and prefix["active_step"] is not None
                assert prefix["active_step"]["library_evaluations_trace"]["pending_record_unsealed"]
                assert prefix["active_step"]["library_evaluations_trace"]["pending_record"]["completed"] is False
                assert prefix["checkpoints"][0]["kind"] == "initial"
                assert np.array_equal(prefix["final_model"], [.8])
    finally:
        bounded.minimize = original_bounded_solver
    assert reads[0] == reads[1] and events[0] == events[1]


def test_separate_model_arrays_keep_old_nonfinite_rejection_without_JSON_expansion():
    for module in (load_old(), bounded):
        clock = Clock()
        def factory(clock=clock):
            return Oracle(clock, [.3])
        arguments = (np.array([.8]), .1, 1., options(), factory)
        ledger = (module._Ledger(*arguments, None, clock) if module is not bounded
                  else module._Ledger(*arguments, clock))
        ledger.initialize()
        ledger.model = np.array([np.inf])
        with pytest.raises(ValueError):
            ledger.finish("failed")


def test_4096_metadata_cardinality_and_fixed_compression_size_envelope():
    # Metadata only: no oracle, optimizer, model arrays, physics or clock.
    physics_keys = ["factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs", "stability_rhs",
                    "dense_inverse_rhs", "svd_evaluations", "cache_hits", "analytic_bound_evaluations"]
    oracle_keys = ["contract_checks", "objective_cache_hits", "objective_evaluations", "objective_contractions",
                   "gradient_evaluations", "gradient_cache_hits", "transpose_symmetry_checks", "factorization_attempts",
                   "operator_assemblies", "failed_factorizations", "pinned_factor_reuses", "linear_solve_dispatches",
                   "adjoint_solve_dispatches", "forward_solve_dispatches"]
    cost = dict(oracle=dict(physics=dict.fromkeys(physics_keys, 999999999),
                           oracle=dict.fromkeys(oracle_keys, 999999999)), proposal={})
    checkpoints, steps, all_library = bounded._Stream(), bounded._Stream(), bounded._Chain()
    checkpoints.append(dict(kind="initial", iteration=-1, elapsed_seconds=1.2345678901234567,
        within_budget=True, objective=1.2345678901234567, model_sha256="f"*64, online_costs=cost))
    for index in range(4096):
        library = bounded._Stream(on_seal=all_library.add)
        for event_index in range(24):
            digest = hashlib.sha256(f"metadata-only-{index}-{event_index}".encode()).hexdigest()
            event = dict(kind="gradient", candidate_sha256=digest, elapsed_start=1.2345678901234567,
                completed=False, gradient_infinity_norm=1.2345678901234567, elapsed_end=2.345678901234567,
                costs=cost)
            library.append(event)
            event["completed"] = True
        library.finalize()
        model_sha = hashlib.sha256(f"metadata-model-{index}".encode()).hexdigest()
        checkpoints.append(dict(kind="accepted", iteration=index, elapsed_seconds=2.345678901234567,
            within_budget=True, objective=1.2345678901234567, model_sha256=model_sha, online_costs=cost))
        steps.append(dict(iteration=index, method="exact_lbfgsb_search", model_sha256=model_sha,
            objective_before=1.2345678901234567, elapsed_start=1.2345678901234567, accepted=True,
            trials=[], exact_recoveries=[], native_callback=dict(candidate_sha256=model_sha,
                objective=1.2345678901234567, strict_decrease_endpoint_eligible=True, elapsed_seconds=2.345678901234567),
            objective_after=1.2345678901234567, accepted_elapsed_seconds=2.345678901234567,
            elapsed_end=2.345678901234567, seconds=1.2345678901234567, online_cost_delta=cost,
            library_evaluations=library.records(), library_evaluations_summary=library.summary()))
        assert len(steps.head)+len(steps.tail)+int(steps.pending is not None) <= 17
    checkpoints.finalize()
    steps.finalize()
    profile = dict(checkpoints=checkpoints.records(), checkpoint_trace=checkpoints.summary(),
        steps=steps.records(), step_trace=steps.summary(), all_library_evaluation_trace=all_library.summary(),
        native_callback_count=4096, native_callback_invocations=4096, native_callback_point_records=4096,
        native_callback_count_semantics="alias of native_callback_invocations; includes entry-guard halts",
        maximum_steps_safety_cap=4096)
    encoded = json.dumps(profile, indent=2, sort_keys=True, allow_nan=False).encode()
    compressed = gzip.compress(encoded, mtime=0)
    assert checkpoints.chain.count == 4097 and steps.chain.count == 4096
    assert all_library.count == 4096*24
    assert len(profile["checkpoints"]) == len(profile["steps"]) == 16
    assert all(len(row["library_evaluations"]) == 16 for row in profile["steps"])
    native_vectors_per_run = 2*2401*8
    report = dict(check="bounded_exact_metadata_size_envelope", project_solves=0, optimizer_calls=0,
        per_stage_profile=dict(callbacks=4096, checkpoints=4097, library_events=4096*24,
                               retained_steps=16, retained_checkpoints=16, retained_library_per_step=16),
        uncompressed_JSON_bytes=len(encoded), gzip_JSON_bytes=len(compressed),
        conservative96_two_stage_engine_metadata_and_two_native_vectors_raw=96*(2*len(encoded)+native_vectors_per_run),
        conservative96_two_stage_engine_metadata_and_two_native_vectors_gzip=96*(2*len(compressed)+native_vectors_per_run),
        proposed_output_guard_bytes=24*1024**2,
        scope="metadata-only conservative cardinality profile, not a reachable optimizer trajectory or rigorous compression bound; runner/header/source/artifact-copy overhead is separate")
    print(json.dumps(report, sort_keys=True))


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
    assert results[1]["native_callback_count"] == results[0]["solver_result"]["nit"]
    assert results[1]["native_callback_point_records"] == sum("native_callback" in step for step in results[0]["steps"])
