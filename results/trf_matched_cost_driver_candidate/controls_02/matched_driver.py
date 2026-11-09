"""Held truth-free matched-cost API; no project loader, CLI, GO or execution route.

Prepare/factory callbacks exist for manufactured lifecycle controls. A future
project caller still requires native integration, source review and root GO.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import optimize
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator

import bounded_exact as engine
from normalized_exact import (
    NormalizedExactOracle,
    count_tree,
    immutable_model,
    model_sha,
)

ARMS = ("data_only_normalized_exact_lbfgsb", "data_only_normalized_cached_j_trf_lsmr")
PROTOCOL_SHA256 = "e967fdf4096ab820b3fa68b37ce711b99dbfd752b3fe5a88e100589a9a30a41e"
COMPONENTS = {
    "bounded_exact.py": "d5fb6326bcd252efc3a7672dba993311fb0ba724b90039e7fda317bcf8bb770c",
    "cached_jacobian.py": "f613370617d7e0bd7815cf441dd8ba5f41816ad4791275e86f33de228657c79b",
    "public_map_builder.py": "7e1c79f99df6e68dda185552f0ff6f480b5fd4823061e7581952cafbd759cb11"}
TRF_OPTIONS = {"method": "trf", "tr_solver": "lsmr", "loss": "linear", "x_scale": 1.,
    "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-10, "max_nfev": 4096,
    "tr_options": {"regularize": True, "damp": 0., "atol": 1e-6, "btol": 1e-6, "conlim": 1e8, "maxiter": None}}


@dataclass(frozen=True)
class PolicyInput:
    initial_model: np.ndarray
    blocks: tuple = ()
    receivers: tuple = ()
    physics: tuple = ()
    binding: tuple = ()
    preparation_costs: object = None


class BudgetHalt(Exception):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


class AbsoluteClock:
    def __init__(self, clock, started):
        self.clock, self.started, self.last, self.first = clock, started, started, True

    def now(self):
        raw = self.clock()
        value = float(raw)
        if isinstance(raw, bool) or not math.isfinite(value) or value < self.last:
            raise ValueError("finite monotone actual clock required")
        self.last = value
        return value

    def age(self):
        return self.now() - self.started

    def guard(self, *, after=False):
        age = self.age()
        if age > 7.5 or (not after and age >= 7.5):
            raise BudgetHalt("absolute optimizer deadline")
        return age

    def engine_clock(self):
        # d5fb's first clock read sets its ledger origin after validation.
        # Returning true t0 once charges all preceding preparation/validation.
        if self.first:
            self.first = False
            return self.started
        return self.now()


def source_admission(directory, protocol):
    """Local metadata only; never reads a project model/input array."""
    directory = Path(directory)
    raw_protocol = (directory/"protocol.json").read_bytes()
    if (hashlib.sha256(raw_protocol).hexdigest() != PROTOCOL_SHA256
            or protocol != json.loads(raw_protocol)):
        raise ValueError("exact held protocol bytes/semantics changed")
    for name, expected in COMPONENTS.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen classical component changed: " + name)
    for name, value in (("bounded_exact.py", engine.run_exact_lbfgsb),
            ("cached_jacobian.py", importlib.import_module("cached_jacobian").CachedJacobianCallbacks),
            ("public_map_builder.py", importlib.import_module("public_map_builder").PublicReceiverMapBuilder)):
        if Path(inspect.getsourcefile(value)).resolve() != (directory/name).resolve():
            raise ValueError("foreign executing classical component: " + name)
    if engine.minimize is not optimize.minimize:
        raise ValueError("unchanged exact engine SciPy minimize was replaced")
    schedule = (directory/"FROZEN48_SCHEDULE.json").read_bytes()
    if (hashlib.sha256(schedule).hexdigest() != protocol["schedule_sha256"]
            or protocol["arms"] != list(ARMS) or protocol["total_runs"] != 48
            or protocol["optimizer_deadline_seconds"] != 7.5 or protocol["complete_return_cap_seconds"] != 8.
            or protocol["trf"] != TRF_OPTIONS or protocol["execution_performed"] is not False):
        raise ValueError("held prospective source/protocol/schedule contract differs")
    rows = json.loads(schedule)
    triples = {(r["case_index"], r["arm"], r["repetition"]) for r in rows}
    if (len(rows) != 48 or triples != {(c, a, k) for c in range(8) for a in ARMS for k in range(3)}
            or any(r["run_index"] != i or r["status"] != "unattempted" for i, r in enumerate(rows))):
        raise ValueError("all48 pre-input slots must remain fixed and unattempted")
    return {"components": dict(COMPONENTS), "schedule_sha256": protocol["schedule_sha256"],
        "actual_public_SciPy_TRF_function": least_squares is optimize.least_squares,
        "project_execution_available": False, "native_v2_prerequisite_pass_is_not_project_GO": True}


def _initial(problem, lower, upper):
    if type(problem) is not PolicyInput:
        raise TypeError("only explicit truth-free PolicyInput is accepted")
    model = immutable_model(problem.initial_model)
    if (not math.isfinite(lower) or not math.isfinite(upper) or not 0 < lower < upper
            or np.any(model <= lower + 1e-10*max(1., abs(lower)))
            or np.any(model >= upper - 1e-10*max(1., abs(upper)))):
        raise ValueError("common direct initializer must be strictly interior; no clipping/perturbation")
    return model


def _safe_prefix(attempt, key, callable_):
    try:
        attempt[key] = callable_()
    except BaseException as exc:  # noqa: BLE001 -- unknown work is not zero
        attempt[key + "_failure"] = {"exception_type": type(exc).__name__, "message": str(exc)}
        attempt["unknown_work"] = True


def execute_one(*, arm, admit, prepare, native_factory, callbacks_factory, attempt,
                lower_bound=1/36, upper_bound=1., absolute_started=None,
                clock=perf_counter, serialize=encoded):
    """Complete measured truth-free API, requiring external observe_return.

    No production inputs or native constructors are supplied by this module.
    Counts are consumed work, never equalized just because wall limits match.
    """
    entered = clock()
    started = entered if absolute_started is None else absolute_started
    if (isinstance(entered, bool) or not math.isfinite(float(entered))
            or isinstance(started, bool) or not math.isfinite(float(started)) or started > entered):
        raise ValueError("finite absolute start required")
    timing = AbsoluteClock(clock, float(started))
    timing.last = float(entered)
    attempt.update(started=float(started), phase="source_admission", unknown_work=False,
        entered=float(entered), pre_entry_seconds=float(entered)-float(started),
        timing_origin="API_entry_only_not_full_process" if absolute_started is None else "supplied_pre_entry_absolute",
        admission_attempted=True, admission_completed=False,
        preparation_attempted=False, preparation_completed=False, preparation_work_unknown=True,
        construction_attempted=False,
        native_constructed=False, callbacks_constructed=False, final_snapshot_attempted=False)
    native = adapter = callbacks = None
    selected = initial = None
    status, error, result = "failed", None, None
    trace, dispatches = [], {}

    def event(kind, **items):
        if len(trace) >= 16384:
            raise ValueError("fixed scalar trace cap exceeded")
        trace.append({"sequence": len(trace), "kind": kind, **items})

    def native_accept(point):
        attempt["native_accept_attempts"] = attempt.get("native_accept_attempts", 0) + 1
        attempt["native_accept_inflight"] = True
        native.accept(point.native_receipt)
        attempt["native_accept_inflight"] = False
        attempt["native_accept_completed"] = attempt.get("native_accept_completed", 0) + 1

    def promote(point, kind):
        nonlocal selected
        previous = selected
        model = immutable_model(point.model)
        identity = model_sha(model)
        native_accept(point)
        age = timing.age()
        event(kind, model_sha256=identity, raw_Phi=float(point.objective), elapsed_seconds=age,
            within_budget=age <= 7.5)
        if age > 7.5:
            if previous is not None:
                native_accept(previous["point"])
            raise BudgetHalt("accepted bookkeeping completed after absolute deadline")
        selected = {"model": model, "model_sha256": identity, "raw_Phi": float(point.objective),
            "elapsed_seconds": age, "point": point}
        attempt["last_eligible"] = {k: v for k, v in selected.items() if k != "point"}

    try:
        if arm not in ARMS:
            raise ValueError("fixed data-only arm required")
        if (lower_bound, upper_bound) != (1/36, 1.):
            raise ValueError("fixed common squared-slowness box required")
        attempt["admission_receipt"] = admit()
        attempt["admission_completed"] = True
        timing.guard()
        attempt["phase"] = "preparation"
        attempt["preparation_attempted"] = True
        # Shared mutable attempt is the preparation failure-prefix sink. The
        # future loader must update it BEFORE every observation dispatch.
        problem = prepare(attempt)
        attempt["preparation_costs"] = count_tree(problem.preparation_costs)
        attempt["preparation_completed"] = True
        attempt["preparation_work_unknown"] = False
        initial = _initial(problem, lower_bound, upper_bound)
        attempt["initial_model"] = initial
        attempt["phase"] = "native_construction"
        timing.guard()

        if arm == ARMS[0]:
            def factory():
                nonlocal native, adapter
                attempt["construction_attempted"] = True
                native = native_factory(problem)
                attempt["native_constructed"] = True
                adapter = NormalizedExactOracle(native, initial)
                attempt["callbacks_constructed"] = True
                return adapter

            options = engine.TrajectoryOptions(budget_seconds=7.5, maximum_steps=4096,
                initial_step=.002, maximum_step=.02, maximum_line_search_trials=12)
            result = engine.run_exact_lbfgsb(oracle_factory=factory, m0=initial,
                lower_bound=lower_bound, upper_bound=upper_bound, options=options,
                maximum_corrections=10, function_tolerance=1e-12, gradient_tolerance=1e-10,
                clock=timing.engine_clock)
            attempt["engine_result"] = result
            status = result["status"]
            points = [p for p in result["checkpoints"] if p["within_budget"] and p["elapsed_seconds"] <= 7.5]
            if points and adapter.current is not None:
                last = points[-1]
                if (model_sha(result["final_model"]) != last["model_sha256"]
                        or adapter.current.model_sha256 != last["model_sha256"]
                        or result["final_objective"] != last["objective"]):
                    raise ValueError("unchanged exact engine did not return its last eligible point")
                selected = {"model": immutable_model(result["final_model"]),
                    "model_sha256": last["model_sha256"], "raw_Phi": adapter.current.raw_objective,
                    "elapsed_seconds": last["elapsed_seconds"]}
                attempt["last_eligible"] = dict(selected)
        else:
            attempt["construction_attempted"] = True
            native = native_factory(problem)
            attempt["native_constructed"] = True
            callbacks = callbacks_factory(native, initial)
            attempt["callbacks_constructed"] = True
            first_query = True

            def residual(model):
                nonlocal first_query
                timing.guard()
                if first_query and not np.array_equal(model, initial):
                    raise ValueError("TRF silently moved direct initializer")
                first_query = False
                attempt["phase"] = "TRF_residual"
                value = callbacks.fun(model)
                age = timing.guard(after=True)
                event("residual_completed", model_sha256=model_sha(model), elapsed_seconds=age)
                if selected is None:
                    promote(callbacks.latest_fun, "initial")
                return value

            def jacobian(model):
                timing.guard()
                attempt["phase"] = "TRF_jacobian_and_maps"
                operator = callbacks.jac(model)
                timing.guard(after=True)

                def product(kind, vector):
                    timing.guard()
                    attempt["phase"] = "TRF_" + kind
                    dispatches[kind + "_attempts"] = dispatches.get(kind + "_attempts", 0) + 1
                    value = getattr(operator, kind)(vector)
                    dispatches[kind + "_completed"] = dispatches.get(kind + "_completed", 0) + 1
                    timing.guard(after=True)
                    return value

                return LinearOperator(operator.shape, dtype=np.float64,
                    matvec=lambda v: product("matvec", v), rmatvec=lambda v: product("rmatvec", v))

            def accepted(intermediate_result):
                timing.guard()
                attempt["phase"] = "TRF_native_callback"
                point = next((p for p in (callbacks.latest_fun, callbacks.latest_jac,
                    None if selected is None else selected.get("point"))
                    if p is not None and np.array_equal(p.model, intermediate_result.x)), None)
                if point is None or not math.isclose(intermediate_result.cost,
                        point.objective/callbacks.phi0, rel_tol=2e-12, abs_tol=2e-14):
                    raise ValueError("native callback does not bind completed residual point/cost")
                if point.objective > selected["raw_Phi"]:
                    raise ValueError("TRF accepted an increasing genuine Phi")
                if point.objective < selected["raw_Phi"]:
                    promote(point, "accepted_callback")
                else:
                    event("equal_callback_no_new_endpoint", elapsed_seconds=timing.guard(after=True))

            result = least_squares(residual, initial.copy(), jac=jacobian,
                bounds=(lower_bound, upper_bound), callback=accepted, **copy.deepcopy(TRF_OPTIONS))
            # result.x is recorded as native termination metadata only; never selected.
            attempt["TRF_result"] = {"status": int(result.status), "success": bool(result.success),
                "message": str(result.message), "nfev": int(result.nfev), "njev": int(result.njev),
                "result_x_sha256_not_selected": model_sha(result.x)}
            status = "solver_success" if result.success else "solver_terminated"
    except BudgetHalt:
        status = "budget_exhausted"
    except Exception as exc:  # noqa: BLE001 -- retain failed run rather than discard or retry
        status, error = "failed", {"exception_type": type(exc).__name__, "message": str(exc)}
    except BaseException as exc:
        attempt.update(phase="interrupted", unknown_work=True,
            engine_pending_prefix=getattr(exc, "bounded_exact_pending_prefix", None))
        raise
    finally:
        if adapter is not None:
            _safe_prefix(attempt, "normalization_prefix", adapter.diagnostics)
        if callbacks is not None:
            _safe_prefix(attempt, "component_prefix", callbacks.cost_snapshot)
        if native is not None:
            attempt["final_snapshot_attempted"] = True
            _safe_prefix(attempt, "final_native_cumulative", lambda: count_tree(native.cost_snapshot()))
        attempt["TRF_guarded_product_dispatches"] = dict(dispatches)
        attempt["TRF_scalar_events"] = trace

    if attempt.get("native_accept_inflight"):
        attempt["unknown_work"] = True
    normalized = None if adapter is None else adapter.diagnostics()
    component = attempt.get("component_prefix")
    known = (not attempt["unknown_work"] and native is not None
        and attempt["preparation_completed"] and attempt["callbacks_constructed"]
        and (normalized is None or not normalized["unknown_work"])
        and (component is None or (not component["native"]["unknown_since_prefix"]
             and not component["map_builder"]["unknown_work_present"])))
    valid = selected is not None and known and status in {
        "solver_success", "solver_terminated", "budget_exhausted", "maximum_steps"}
    phi0 = None if adapter is None and callbacks is None else (
        adapter.phi0 if adapter is not None else callbacks.phi0)
    payload = {"arm": arm, "status": status, "error": error, "policy_valid": valid,
        "selected": None if selected is None else {k: v for k, v in selected.items() if k not in {"point", "model"}},
        "Phi0": phi0, "common_Phi_ratio": None if selected is None else selected["raw_Phi"]/phi0,
        "initial_model": None if initial is None else initial.tolist(),
        "selected_model": None if selected is None else selected["model"].tolist(),
        "normalization_prefix": normalized, "component_prefix": component,
        "native_cumulative_once": attempt.get("final_native_cumulative"),
        "preparation_costs_disjoint": attempt.get("preparation_costs"),
        "preparation_prefix": attempt.get("preparation_prefix"),
        "preparation_work_unknown": attempt["preparation_work_unknown"],
        "admission_receipt": attempt.get("admission_receipt"),
        "TRF_scalar_events": trace, "TRF_guarded_product_dispatches": dispatches,
        "TRF_result_not_used_for_selection": attempt.get("TRF_result"),
        "exact_engine": None if arm != ARMS[0] or result is None else {
            k: v for k, v in result.items() if k not in {"initial_model", "final_model"}},
        "unknown_work": not known, "scientific_goals_complete": False,
        "truth_used_for_policy": False, "LSMR_actual_iterations_unavailable": None,
        "timing_origin": attempt["timing_origin"], "pre_entry_seconds": attempt["pre_entry_seconds"],
        "complete_OS_process_eligibility_available": False}
    attempt.update(phase="complete_payload_serialization", return_payload_before_serialization=payload)
    serialized = serialize(payload)
    if not isinstance(serialized, bytes):
        raise TypeError("complete serializer must return bytes")
    envelope = {"started": float(started), "payload": payload, "serialized": serialized,
        "selected_model": None if selected is None else selected["model"], "initial_model": initial}
    envelope["inner_return_age"] = timing.age()
    attempt["phase"] = "returned"
    return envelope


def observe_return(envelope, returned):
    """Mandatory immediate external caller timestamp, after full API returns."""
    age = float(returned) - envelope["started"]
    if isinstance(returned, bool) or not math.isfinite(age) or not 0 <= envelope["inner_return_age"] <= age:
        raise ValueError("invalid observed complete-return clock")
    return {"actual_complete_return_seconds": age, "cap_seconds": 8.,
        "eligible": envelope["payload"]["policy_valid"] is True and age <= 8.,
        "eligibility_scope": "Observed API return; future actual OS-terminal controller must gate complete-process eligibility separately."}


if __name__ == "__main__":
    print(json.dumps({"status": "HELD_API_ONLY", "project_execution_available": False,
        "native_v2_prerequisite_pass_is_not_project_GO": True, "GO_created": False, "project_PDE_calls": 0}))
