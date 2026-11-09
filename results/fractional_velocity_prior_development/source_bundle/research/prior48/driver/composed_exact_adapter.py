"""Classical composed-objective adapter for the byte-frozen bounded engine.

Only supplied initial m and native data-objective operations are available.
Composed points are distinct from native merit receipts and are not certificates.
This module supplies no data loader, PDE, study schedule or execution permission.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import math
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE_SHA256 = "d5fb6326bcd252efc3a7672dba993311fb0ba724b90039e7fda317bcf8bb770c"
PRIOR_SHA256 = "4aa248fd3862533390ad918f4ad3f70d02fcdebcdc8d784494d375c935536163"
_MODULE_SEQUENCE = itertools.count()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha(model):
    return hashlib.sha256(np.ascontiguousarray(model, dtype="<f8").tobytes()).hexdigest()


def immutable(values):
    raw = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(raw.tobytes(), dtype=np.float64).reshape(raw.shape)


@dataclass(frozen=True)
class FrozenDependencies:
    engine: object
    prior: object
    engine_path: Path
    prior_path: Path

    def validate(self):
        for module, path, expected, function in (
            (self.engine, self.engine_path, ENGINE_SHA256, self.engine.run_exact_lbfgsb),
            (self.prior, self.prior_path, PRIOR_SHA256, self.prior.FractionalVelocityPrior),
        ):
            if (Path(module.__file__).resolve() != path or file_sha(path) != expected
                    or Path(inspect.getsourcefile(function)).resolve() != path):
                raise ValueError("frozen dependency source/origin mismatch")


def load_frozen_dependencies():
    modules = []
    for filename, expected in (("bounded_exact.py", ENGINE_SHA256), ("fractional_velocity_prior.py", PRIOR_SHA256)):
        path = (HERE / filename).resolve()
        if file_sha(path) != expected:
            raise ValueError("frozen dependency bytes changed")
        name = f"_composed_exact_private_{next(_MODULE_SEQUENCE)}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    dependencies = FrozenDependencies(*modules, (HERE / "bounded_exact.py").resolve(), (HERE / "fractional_velocity_prior.py").resolve())
    dependencies.validate()
    return dependencies


@dataclass(frozen=True, eq=False)
class ComposedObjectivePoint:
    model: np.ndarray = field(repr=False)
    objective: float
    normalized_data_value: float
    weighted_prior_value: float
    # This is the actual delegate-issued receipt, never a manufactured native one.
    native_receipt: object = field(repr=False)
    prior_evaluation: object = field(repr=False)


@dataclass(frozen=True)
class ComposedGradient:
    gradient: np.ndarray
    point: ComposedObjectivePoint


class ComposedObjectiveOracle:
    """Current/latest composed-point cache over one actual native owner.

    Every engine evaluate/gradient request still calls the native delegate.
    Prior results are cached only when that delegate reissues an identical
    current/latest native receipt; no native query is suppressed or prepaid.
    """

    def __init__(self, native, *, initial_model, shape, grid_spacing, smoothing_length, weight, prior_module):
        if (len(shape) != 2 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 1 for x in shape)):
            raise ValueError("two positive grid dimensions required")
        initial = np.asarray(initial_model)
        if (initial.dtype != np.float64 or initial.shape != (math.prod(shape),)
                or not np.isfinite(initial).all() or (initial <= 0).any()):
            raise ValueError("direct positive finite flat float64 initial m required")
        if isinstance(weight, (bool, np.bool_)) or weight not in (0.0, 100.0):
            raise ValueError("only separately fixed lambda0 and lambda100 arms are supported")
        if native.n != initial.size:
            raise ValueError("native owner dimension differs")
        self.native, self.n, self.shape, self.weight = native, initial.size, tuple(map(int, shape)), float(weight)
        self.initial_model = immutable(initial)
        self.initial_model_sha256 = model_sha(self.initial_model)
        self.initial_data_value = None
        self.current = self.trial = None
        self.first_accepted_header = self.current_accepted_header = None
        self._issued = weakref.WeakKeyDictionary()
        self.last_native_snapshot = None
        self.counts = {"constructions": 1, "initial_model_copy_bytes": initial.nbytes,
            "prior_construction_attempts": 1, "prior_constructions_completed": 0,
            "evaluate_attempts": 0, "native_evaluate_dispatches": 0, "native_evaluate_completed": 0,
            "gradient_attempts": 0, "native_gradient_dispatches": 0, "native_gradient_completed": 0,
            "accept_attempts": 0, "native_accept_dispatches": 0, "native_accept_completed": 0,
            "prior_evaluation_attempts": 0, "prior_evaluations_completed": 0, "composed_point_cache_hits": 0,
            "normalization_initializations": 0, "objective_scalings": 0, "gradient_compositions": 0,
            "composed_points_issued": 0, "point_validation_attempts": 0, "model_fingerprint_bytes": 0,
            "prior_fingerprint_bytes": 0,
            "accepted_header_exports": 0, "cost_snapshot_calls": 0, "prior_gradient_copy_bytes": 0}
        # Both arms pay this construction; lambda0 performs no prior evaluation.
        self.prior = prior_module.FractionalVelocityPrior(self.initial_model.reshape(self.shape), grid_spacing, smoothing_length)
        self.counts["prior_constructions_completed"] += 1

    def _vector(self, model):
        value = np.asarray(model)
        if (value.dtype != np.float64 or value.shape != (self.n,) or not np.isfinite(value).all()
                or (value <= 0).any()):
            raise ValueError("requested m must be a positive finite native float64 vector")
        return value

    def _fingerprint(self, point):
        self.counts["model_fingerprint_bytes"] += point.model.nbytes
        prior = point.prior_evaluation
        prior_fingerprint = None
        if prior is not None:
            self.counts["prior_fingerprint_bytes"] += prior.gradient.nbytes
            prior_fingerprint = (float(prior.value), float(prior.zeroth_order),
                                 float(prior.roughness), model_sha(prior.gradient))
        return (model_sha(point.model), float(point.objective), float(point.normalized_data_value),
                float(point.weighted_prior_value), id(point.native_receipt), prior_fingerprint)

    def _check(self, point):
        self.counts["point_validation_attempts"] += 1
        if type(point) is not ComposedObjectivePoint or point not in self._issued:
            raise ValueError("foreign or fabricated composed point")
        if self._fingerprint(point) != self._issued[point]:
            raise ValueError("composed point fingerprint changed")
        if not np.array_equal(point.model, point.native_receipt.model):
            raise ValueError("native and composed model differ")

    def evaluate(self, model):
        self.counts["evaluate_attempts"] += 1
        model = self._vector(model)
        if self.initial_data_value is None and model_sha(model) != self.initial_model_sha256:
            raise ValueError("first paid query must use the exact supplied initial m")
        self.counts["native_evaluate_dispatches"] += 1
        native = self.native.evaluate(model)
        self.counts["native_evaluate_completed"] += 1
        data_value = float(native.objective)
        if (not math.isfinite(data_value) or data_value < 0
                or np.asarray(native.model).dtype != np.float64 or not np.array_equal(native.model, model)):
            raise ValueError("delegate native data receipt/model differs")
        if self.initial_data_value is None:
            if data_value <= 0:
                raise ValueError("first paid query must be initial native m with positive Phi0")
            self.initial_data_value = data_value
            self.counts["normalization_initializations"] += 1
        for point in (self.current, self.trial):
            if point is not None and point.native_receipt is native:
                self._check(point)
                self.counts["composed_point_cache_hits"] += 1
                return point
        prior_result = None
        weighted = 0.0
        if self.weight:
            self.counts["prior_evaluation_attempts"] += 1
            prior_result = self.prior.evaluate(model.reshape(self.shape))
            self.counts["prior_evaluations_completed"] += 1
            # Freeze the supplied prior gradient once; no extra data gradient.
            prior_result = type(prior_result)(prior_result.value, prior_result.zeroth_order,
                                             prior_result.roughness, immutable(prior_result.gradient))
            self.counts["prior_gradient_copy_bytes"] += prior_result.gradient.nbytes
            weighted = self.weight * prior_result.value
        self.counts["objective_scalings"] += 1
        normalized = data_value / self.initial_data_value
        value = normalized + weighted
        if not math.isfinite(value):
            raise FloatingPointError("composed objective is not finite")
        point = ComposedObjectivePoint(immutable(model), value, normalized, weighted, native, prior_result)
        self._issued[point] = self._fingerprint(point)
        self.counts["composed_points_issued"] += 1
        self.trial = point
        return point

    def gradient(self, point):
        self.counts["gradient_attempts"] += 1
        self._check(point)
        self.counts["native_gradient_dispatches"] += 1
        result = self.native.gradient(point.native_receipt)
        self.counts["native_gradient_completed"] += 1
        gradient = np.asarray(result.gradient)
        if gradient.dtype != np.float64 or gradient.shape != (self.n,) or not np.isfinite(gradient).all():
            raise ValueError("invalid native data gradient")
        self.counts["gradient_compositions"] += 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            combined = gradient / self.initial_data_value
            if self.weight:
                combined = combined + self.weight * point.prior_evaluation.gradient.reshape(-1)
        if not np.isfinite(combined).all():
            raise FloatingPointError("composed gradient is not finite")
        return ComposedGradient(immutable(combined), point)

    def accept(self, point):
        self.counts["accept_attempts"] += 1
        self._check(point)
        self.counts["native_accept_dispatches"] += 1
        # Native accept verifies owner/issued receipt/model authorization only;
        # it does not impose a raw data-objective descent gate.
        self.native.accept(point.native_receipt)
        self.counts["native_accept_completed"] += 1
        self.current, self.trial = point, None
        self.counts["accepted_header_exports"] += 1
        header = {"kind": "classical_composed_objective_point", "model_sha256": model_sha(point.model),
            "objective": point.objective, "normalized_data_value": point.normalized_data_value,
            "weighted_prior_value": point.weighted_prior_value,
            "native_data_receipt_header": dict(point.native_receipt.to_dict()),
            "native_receipt_fabricated": False, "certificate_claim": False}
        self.current_accepted_header = header
        if self.first_accepted_header is None:
            self.first_accepted_header = dict(header)

    def cost_snapshot(self):
        self.counts["cost_snapshot_calls"] += 1
        native = self.native.cost_snapshot()
        self.last_native_snapshot = native
        return {"native": native, "adapter": dict(self.counts)}

    def diagnostics_no_queries(self):
        return {"initial_native_m_sha256": self.initial_model_sha256, "initial_paid_Phi0": self.initial_data_value,
            "weight": self.weight, "initial_accepted": self.first_accepted_header, "current_accepted": self.current_accepted_header,
            "adapter_counts": dict(self.counts), "latest_native_snapshot": self.last_native_snapshot,
            "scope": "native data-query counters and adapter arithmetic are separate; no certificate claim"}


class GlobalAnchorClock:
    """Only the engine's first clock read returns the caller's true t0."""
    def __init__(self, clock, started):
        if isinstance(started, (bool, np.bool_)) or not math.isfinite(started):
            raise ValueError("finite true global start required")
        self.clock, self.started, self.last, self.calls = clock, float(started), float(started), 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            return self.started
        raw = self.clock()
        now = float(raw)
        if isinstance(raw, (bool, np.bool_)) or not math.isfinite(now) or now < self.last:
            raise ValueError("clock must be finite and globally monotone")
        self.last = now
        return now


def run_composed_exact(*, native_oracle_factory, initial_model, shape, grid_spacing,
                       smoothing_length, weight, dependencies, options, global_started, clock,
                       lower_bound=1/36, upper_bound=1., maximum_corrections=10,
                       function_tolerance=1e-12, gradient_tolerance=1e-10):
    """One engine call, without a study runner or post-return truth metrics.

    Caller starts t0 before preparation and records actual return after the
    complete payload. Dependency import bootstrap may be prepared separately;
    this call's source checks, factory/prior construction and arithmetic remain
    inside the true global age. Engine's retained last-eligible selection and
    global deadline are unchanged. Returned engine seconds exclude this final
    wrapper metadata assembly; caller-observed total return is authoritative.
    """
    dependencies.validate()
    owners = []
    construction = {"native_factory_attempted": False, "native_factory_returned": False,
                        "adapter_constructor_attempted": False, "adapter_constructor_completed": False,
                        "creation_in_flight_costs_unknown": False}

    def factory():
        construction["native_factory_attempted"] = True
        construction["creation_in_flight_costs_unknown"] = True
        native = native_oracle_factory()
        construction["native_factory_returned"] = True
        construction["adapter_constructor_attempted"] = True
        owner = ComposedObjectiveOracle(native, initial_model=initial_model, shape=shape,
            grid_spacing=grid_spacing, smoothing_length=smoothing_length, weight=weight,
            prior_module=dependencies.prior)
        owners.append(owner)
        construction["adapter_constructor_completed"] = True
        construction["creation_in_flight_costs_unknown"] = False
        return owner

    anchor = GlobalAnchorClock(clock, global_started)
    try:
        result = dependencies.engine.run_exact_lbfgsb(oracle_factory=factory, m0=initial_model,
            lower_bound=lower_bound, upper_bound=upper_bound, options=options,
            maximum_corrections=maximum_corrections, function_tolerance=function_tolerance,
            gradient_tolerance=gradient_tolerance, clock=anchor)
    except BaseException as exc:
        exc.composed_adapter_prefix = None if not owners else owners[0].diagnostics_no_queries()
        if exc.composed_adapter_prefix is not None:
            exc.composed_adapter_prefix.update(in_flight_native_work_unknown=True,
                                               completed_return_claim=False)
        exc.composed_construction_prefix = dict(construction)
        raise
    result["acceptance"] = "native L-BFGS-B; strict decrease of F_lambda required for new eligible endpoint; raw Phi may increase"
    result["certificate_scope"] = None
    result["certificate_claim"] = False
    result["objective_scope"] = "F_lambda=Phi/Phi0+lambda*R; all engine objective events use this composed scalar"
    result["adapter"] = None if not owners else owners[0].diagnostics_no_queries()
    result["construction_prefix"] = dict(construction)
    result["dependency_sources"] = {"engine_sha256": ENGINE_SHA256, "prior_sha256": PRIOR_SHA256,
                                        "byte_identical_copies": True}
    result["global_clock_anchor_calls"] = anchor.calls
    result["actual_complete_return_must_be_stamped_by_caller"] = True
    return result
