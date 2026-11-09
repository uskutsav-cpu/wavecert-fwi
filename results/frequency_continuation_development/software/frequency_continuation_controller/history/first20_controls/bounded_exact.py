"""Bounded-history instrumentation of the frozen paid exact L-BFGS-B path.

Standard SciPy optimizer and oracle semantics are unchanged. Checkpoints retain
only scalar/hash/counter events; initial/current/final vectors are O(n), not
O(steps*n). This is AI-assisted measurement code, not a new optimization method.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

ORIGINAL_PAID_ENGINE_SHA256 = "b863af87f03c2470fde8d72412b1c89e76e389a528cb7d8b08efe89ceeb3a12c"
# Unpassed, unchanged SciPy _minimize_lbfgsb default; no claim of unbounded work.
SCIPY_MAXFUN_DEFAULT = 15000

@dataclass(frozen=True)
class TrajectoryOptions:
    budget_seconds: float
    maximum_steps: int
    initial_step: float
    maximum_step: float
    maximum_line_search_trials: int = 12
    contraction: float = .5
    accepted_step_growth: float = 2.
    relative_objective_target: float | None = None
    absolute_objective_target: float | None = None
    exact_projected_stationarity_tolerance: float = 1e-10

class _Halt(Exception):
    def __init__(self, status):
        self.status = status

def _immutable(values):
    values = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)

def _digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value

def _strict_metadata(value):
    result = _jsonable(value)
    json.dumps(result, allow_nan=False)
    return result

def _counts(value):
    """Counter-only contract; dimensions/provenance belong in hook metadata."""
    if not isinstance(value, dict):
        raise TypeError("cost_snapshot must return a nested dictionary of counts")
    result = {}
    for key, item in value.items():
        if key in {"dimensions", "full_dimension", "reduced_dimension"}:
            continue
        if isinstance(item, dict):
            result[key] = _counts(item)
        elif isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)) and item >= 0:
            result[key] = int(item)
        else:
            raise ValueError(f"cost_snapshot contains a non-count value: {key}")
    return result

def _delta(before, after):
    result = {}
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key, 0), after.get(key, 0)
        if isinstance(a, dict) or isinstance(b, dict):
            result[key] = _delta(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {})
        else:
            result[key] = b-a
            if result[key] < 0:
                raise ValueError("online work counters cannot decrease during a run")
    return result

def _validate(m0, lower, upper, options):
    model = np.asarray(m0)
    if model.ndim != 1 or np.iscomplexobj(model) or not model.size or not np.all(np.isfinite(model)):
        raise ValueError("initial model must be a nonempty finite real vector")
    lo, hi = np.broadcast_to(lower, model.shape), np.broadcast_to(upper, model.shape)
    if (np.iscomplexobj(lo) or np.iscomplexobj(hi) or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi)) or np.any(lo > hi) or np.any(model < lo) or np.any(model > hi)):
        raise ValueError("initial model must satisfy finite ordered box constraints")
    for name in ("budget_seconds", "initial_step", "maximum_step"):
        if not np.isfinite(getattr(options, name)) or getattr(options, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("maximum_steps", "maximum_line_search_trials"):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "maximum_steps" else 1):
            raise ValueError(f"{name} has an invalid count")
    if (options.maximum_step < options.initial_step or not 0 < options.contraction < 1
            or not np.isfinite(options.accepted_step_growth) or options.accepted_step_growth < 1
            or not np.isfinite(options.exact_projected_stationarity_tolerance)
            or options.exact_projected_stationarity_tolerance < 0):
        raise ValueError("invalid step schedule or exact stationarity tolerance")
    if options.relative_objective_target is not None and not 0 <= options.relative_objective_target < 1:
        raise ValueError("relative objective target must be in [0, 1)")
    if options.absolute_objective_target is not None and (
            not np.isfinite(options.absolute_objective_target) or options.absolute_objective_target < 0):
        raise ValueError("absolute objective target must be finite and nonnegative")
    return _immutable(model), _immutable(lo), _immutable(hi)

class _Ledger:
    def __init__(self, m0, lower, upper, options, oracle_factory, clock):
        self.model, self.lower, self.upper = _validate(m0, lower, upper, options)
        self.initial_model = self.model
        self.options, self.clock = options, clock
        self.oracle_factory = oracle_factory
        self.oracle = self.current = None
        self.checkpoints, self.steps, self.active = [], [], None
        self.initial_objective = self.setup_seconds = self.initial_merit_seconds = None
        self.setup_costs = {}
        self.message, self.solver_result = None, None
        self.started = clock()

    def elapsed(self):
        elapsed = float(self.clock()-self.started)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("trajectory clock must be finite and monotone")
        return elapsed

    def guard(self, *, after=False):
        elapsed = self.elapsed()
        if elapsed > self.options.budget_seconds or (not after and elapsed >= self.options.budget_seconds):
            raise _Halt("budget_exhausted")
        return elapsed

    def snapshot(self):
        return {"oracle": {} if self.oracle is None else _counts(self.oracle.cost_snapshot()),
                "proposal": {}}

    def initialize(self):
        self.oracle = self.oracle_factory()
        self.setup_seconds = self.elapsed()
        self.setup_costs = self.snapshot()
        self.guard()
        start = self.elapsed()
        receipt = self.oracle.evaluate(self.model)
        self._validate_receipt(receipt, self.model)
        self.oracle.accept(receipt)
        self.current, self.initial_objective = receipt, float(receipt.objective)
        time = self.elapsed()
        self.initial_merit_seconds = time-start
        self.checkpoints.append(dict(kind="initial", iteration=-1, elapsed_seconds=time,
            within_budget=time <= self.options.budget_seconds, objective=self.initial_objective,
            model_sha256=_digest(self.model), online_costs=self.snapshot()))
        self.guard(after=True)
        self.target_guard()

    def target_guard(self):
        targets = []
        if self.options.absolute_objective_target is not None:
            targets.append(self.options.absolute_objective_target)
        if self.options.relative_objective_target is not None:
            if self.initial_objective < 0:
                raise ValueError("relative objective targets require a nonnegative initial objective")
            targets.append(self.options.relative_objective_target*self.initial_objective)
        if targets and self.current.objective <= max(targets):
            raise _Halt("objective_target")

    def _validate_receipt(self, receipt, model):
        if (not np.isfinite(receipt.objective) or np.asarray(receipt.model).shape != self.model.shape
                or not np.array_equal(receipt.model, model)):
            raise ValueError("paid merit receipt does not match the requested finite model/objective")


    def begin_step(self, iteration, method):
        self.guard()
        self.active = dict(iteration=iteration, method=method, model_sha256=_digest(self.model),
            objective_before=float(self.current.objective), elapsed_start=self.elapsed(),
            costs_before=self.snapshot(), accepted=False, trials=[], exact_recoveries=[])

    def end_step(self):
        if self.active is not None:
            self.active["elapsed_end"] = self.elapsed()
            self.active["seconds"] = self.active["elapsed_end"]-self.active["elapsed_start"]
            self.active["online_cost_delta"] = _delta(self.active.pop("costs_before"), self.snapshot())
            self.steps.append(self.active)
            self.active = None

    def commit(self, receipt, *, iteration):
        previous = self.current
        next_model = _immutable(receipt.model)
        self.oracle.accept(receipt)
        time = self.elapsed()
        if time > self.options.budget_seconds:
            # Promotion is bookkeeping, not permission to credit a post-budget
            # state. Restore the owned current receipt and retain all costs.
            self.oracle.accept(previous)
            raise _Halt("budget_exhausted")
        self.current, self.model = receipt, next_model
        self.checkpoints.append(dict(kind="accepted", iteration=iteration, elapsed_seconds=time,
            within_budget=True, objective=float(receipt.objective),
            model_sha256=_digest(self.model), online_costs=self.snapshot()))
        if self.active is not None:
            self.active["accepted"] = True
            self.active["objective_after"] = float(receipt.objective)
            self.active["accepted_elapsed_seconds"] = time

    def finish(self, status, error=None):
        if self.active is not None:
            self.active["termination"] = status
            if error is not None:
                self.active["error"] = f"{type(error).__name__}: {error}"
        self.end_step()
        costs = self.snapshot()
        elapsed = self.elapsed()
        result = dict(status=status, message=None if error is None else f"{type(error).__name__}: {error}",
            initial_model=self.initial_model, final_model=self.model,
            final_objective=None if self.current is None else float(self.current.objective),
            initial_objective=self.initial_objective, checkpoints=tuple(self.checkpoints), steps=tuple(self.steps),
            setup_seconds=self.setup_seconds, initial_merit_seconds=self.initial_merit_seconds,
            total_online_seconds=elapsed, budget_seconds=self.options.budget_seconds,
            budget_overshoot_seconds=max(0., elapsed-self.options.budget_seconds),
            setup_costs=self.setup_costs, total_online_costs=costs,
            costs_complete=self.oracle is not None,
            solver_result=self.solver_result,
            acceptance="paid accurate-reference objective strict decrease; ordinary floating point",
            certificate_scope="raw proposed direction only; clipping can change it; finite steps require paid merit",
            truth_used_for_policy=False, offline_validation_work_performed=False,
            checkpoint_vectors_retained=False,
            native_callback_count=sum("native_callback" in step for step in self.steps),
            maximum_steps_safety_cap=self.options.maximum_steps,
            scipy_maxfun_default=SCIPY_MAXFUN_DEFAULT, scipy_maxfun_passed_explicitly=False)
        # Vectors remain separately retained, immutable O(n) arrays. Never
        # expand checkpoint vectors or native model arrays into JSON here.
        _strict_metadata({k: v for k, v in result.items()
                          if k not in {"initial_model", "final_model"}})
        return result

def run_exact_lbfgsb(*, oracle_factory, m0, lower_bound, upper_bound, options: TrajectoryOptions,
                     maximum_corrections=10, function_tolerance=1e-12, gradient_tolerance=1e-10,
                     clock=perf_counter):
    """Strong exact comparator using the same paid merit/gradient receipt cache.

    Only completed native callbacks with strict fine-objective decrease create
    budget-eligible endpoints. Equal native callbacks remain paid events and
    do not alter the native algorithm. Trials completed after budget receive
    no endpoint credit. Native termination is preserved, not relabeled by us.
    """
    if (isinstance(maximum_corrections, bool) or not isinstance(maximum_corrections, int) or maximum_corrections < 1
            or not np.isfinite(function_tolerance) or function_tolerance < 0
            or not np.isfinite(gradient_tolerance) or gradient_tolerance < 0):
        raise ValueError("invalid native L-BFGS-B options")
    ledger = _Ledger(m0, lower_bound, upper_bound, options, oracle_factory, clock)
    status, error = "solver_terminated", None
    try:
        ledger.initialize()
        if options.maximum_steps == 0:
            raise _Halt("maximum_steps")
        iteration = 0

        def begin_event(model, kind):
            if ledger.active is None:
                ledger.begin_step(iteration, "exact_lbfgsb_search")
            event = dict(kind=kind, candidate_sha256=_digest(model), elapsed_start=ledger.elapsed(),
                         completed=False)
            ledger.active.setdefault("library_evaluations", []).append(event)
            return event

        def function(model):
            ledger.guard()
            event = begin_event(model, "merit")
            receipt = ledger.oracle.evaluate(model)
            ledger._validate_receipt(receipt, model)
            event.update(completed=True, objective=float(receipt.objective), elapsed_end=ledger.elapsed(),
                         costs=ledger.snapshot())
            ledger.guard(after=True)
            return float(receipt.objective)

        def jacobian(model):
            ledger.guard()
            event = begin_event(model, "gradient")
            receipt = ledger.oracle.evaluate(model)
            result = ledger.oracle.gradient(receipt)
            gradient = np.asarray(result.gradient)
            if gradient.shape != ledger.model.shape or np.iscomplexobj(gradient) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("exact L-BFGS-B gradient is nonfinite or has invalid shape")
            event.update(completed=True, gradient_infinity_norm=float(np.max(np.abs(gradient))),
                         elapsed_end=ledger.elapsed(), costs=ledger.snapshot())
            ledger.guard(after=True)
            return gradient

        def callback(model):
            nonlocal iteration
            ledger.guard()
            if ledger.active is None:
                ledger.begin_step(iteration, "exact_lbfgsb_search")
            receipt = ledger.oracle.evaluate(model)
            ledger._validate_receipt(receipt, model)
            if receipt.objective > ledger.current.objective:
                raise RuntimeError("L-BFGS-B accepted an increasing fine objective")
            strict_decrease = receipt.objective < ledger.current.objective
            completed = ledger.elapsed()
            ledger.active["native_callback"] = dict(candidate_sha256=_digest(model),
                objective=float(receipt.objective),
                strict_decrease_endpoint_eligible=strict_decrease and completed <= options.budget_seconds,
                elapsed_seconds=completed)
            ledger.guard(after=True)
            if strict_decrease:
                ledger.commit(receipt, iteration=iteration)
            else:
                ledger.active["reason"] = "equal native objective; paid callback without new endpoint credit"
            ledger.end_step()
            iteration += 1
            ledger.target_guard()

        result = minimize(function, ledger.model.copy(), jac=jacobian, method="L-BFGS-B",
            bounds=list(zip(ledger.lower, ledger.upper, strict=True)), callback=callback,
            options=dict(maxiter=options.maximum_steps, maxls=options.maximum_line_search_trials,
                maxcor=maximum_corrections, ftol=function_tolerance, gtol=gradient_tolerance))
        ledger.solver_result = dict(success=bool(result.success), status=int(result.status),
            message=str(result.message), nit=int(result.nit), nfev=int(result.nfev), njev=int(result.njev))
        status = "solver_success" if result.success else "solver_terminated"
    except _Halt as halt:
        status = halt.status
    except Exception as exc:
        status, error = "failed", exc
    except BaseException as exc:
        # Preserve interruption propagation and attach the already-paid prefix.
        # A controller can retain this without inventing a completed return.
        try:
            exc.bounded_exact_prefix = ledger.finish("interrupted", exc)
        except Exception as prefix_error:
            exc.bounded_exact_prefix_error = f"{type(prefix_error).__name__}: {prefix_error}"
        raise
    output = ledger.finish(status, error)
    output["acceptance"] = "native exact L-BFGS-B; strict fine-objective decrease required for new endpoint credit; equal callbacks remain paid events"
    return output
