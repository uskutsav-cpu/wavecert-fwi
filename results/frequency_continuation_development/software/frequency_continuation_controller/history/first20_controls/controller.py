"""Provisional paid native-m baseline controller; no project runner or truth API.

Only execute_timed is the measured policy API. Its caller records actual return
after the full payload, counters, model hashes and strict metadata are ready.
Absolute phase ceilings are not equal actual durations. This is standard
baseline software assistance, not a student scientific design or new method.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

ALL8 = (.25, .5, 1., 2., 2.5, 3., 3.5, 4.)
LOW4, ORIGINAL4 = ALL8[:4], ALL8[4:]
ARMS = {"joint8": (ALL8,), "low4_then8": (LOW4, ALL8),
        "restart8": (ALL8, ALL8), "original4": (ORIGINAL4,)}
PRODUCTION_SHA256 = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
SOURCE_PINS = {
    "physics/exact_merit.py": "0fec1f215b28e36017c1e28409d7f7e909a03cbf5b717de5d23eb8d52eb26810",
    "physics/helmholtz.py": "caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131",
    "optimization/paid_descent.py": "b863af87f03c2470fde8d72412b1c89e76e389a528cb7d8b08efe89ceeb3a12c",
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _freeze(values, dtype="<f8"):
    a = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _counts(value):
    if not isinstance(value, dict):
        raise TypeError("a cost provider must return a nested count dictionary")
    result = {}
    for key, v in value.items():
        if isinstance(v, dict):
            result[str(key)] = _counts(v)
        elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 0:
            result[str(key)] = int(v)
        else:
            raise ValueError("invalid count (booleans are not work): " + str(key))
    return result


def _monotone(previous, current):
    for key in set(previous) | set(current):
        a, b = previous.get(key, 0), current.get(key, 0)
        if isinstance(a, dict) or isinstance(b, dict):
            if not isinstance(a, dict) and a != 0 or not isinstance(b, dict) and b != 0:
                raise ValueError("counter tree changed type")
            _monotone(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {})
        elif b < a:
            raise ValueError("cumulative cost counter decreased")


def _sum_counts(items):
    result = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, dict):
                result[key] = _sum_counts([result.get(key, {}), value])
            else:
                if isinstance(result.get(key), dict):
                    raise ValueError("incompatible counter trees")
                result[key] = result.get(key, 0) + value
    return result


@dataclass(frozen=True)
class ControllerSettings:
    engine_deadline_seconds: float = 7.5
    first_stage_deadline_seconds: float = 3.75
    api_return_cap_seconds: float = 8.
    single_stage_maximum_steps: int = 4096
    per_stage_maximum_steps: int = 2048
    maximum_corrections: int = 10
    maximum_line_search_trials: int = 12
    function_tolerance: float = 1e-12
    gradient_tolerance: float = 1e-10

    def validate(self):
        a, b, c = self.first_stage_deadline_seconds, self.engine_deadline_seconds, self.api_return_cap_seconds
        if any(isinstance(v, (bool, np.bool_)) or not math.isfinite(v) for v in (a, b, c)) or not 0 < a < b < c:
            raise ValueError("require positive absolute phase ceilings first<engine<return")
        for key in ("single_stage_maximum_steps", "per_stage_maximum_steps", "maximum_corrections", "maximum_line_search_trials"):
            v = getattr(self, key)
            minimum = 0 if "maximum_steps" in key else 1
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError("invalid integer safety cap: " + key)
        for key in ("function_tolerance", "gradient_tolerance"):
            if isinstance(getattr(self, key), (bool, np.bool_)) or not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError("invalid native stopping tolerance: " + key)


@dataclass(frozen=True)
class PreparedProblem:
    physics: dict
    initial_model: np.ndarray
    blocks: tuple
    receiver_indices: tuple


@dataclass(frozen=True)
class NativeAPI:
    physics_type: type
    block_type: type
    oracle_type: type
    options_type: type
    engine_run: object
    package_root: Path
    source_binding: dict


def load_native_api(repo, *, engine_run, engine_source_path, engine_source_sha256):
    """Explicit pinned engine dependency; no fallback or monkeypatching."""
    repo = Path(repo).resolve()
    package = repo / "src/wavecert"
    for name, expected in SOURCE_PINS.items():
        if file_sha(package / name) != expected:
            raise ValueError("frozen source changed: " + name)
    sys.path.insert(0, str(repo / "src"))
    exact = importlib.import_module("wavecert.physics.exact_merit")
    physics = importlib.import_module("wavecert.physics.helmholtz")
    engine = importlib.import_module("wavecert.optimization.paid_descent")
    provenance = importlib.import_module("wavecert.provenance")
    for module, name in [(exact, "physics/exact_merit.py"), (physics, "physics/helmholtz.py"),
                         (engine, "optimization/paid_descent.py")]:
        if Path(module.__file__).resolve() != package / name:
            raise ValueError("wrong installed module origin: " + name)
    if provenance.source_digest(repo) != PRODUCTION_SHA256:
        raise ValueError("production9f manifest changed")
    engine_source_path = Path(engine_source_path).resolve()
    if (Path(inspect.getsourcefile(engine_run)).resolve() != engine_source_path
            or file_sha(engine_source_path) != engine_source_sha256):
        raise ValueError("explicit engine source/origin pin mismatch")
    return NativeAPI(physics.Helmholtz2D, physics.SurveyBlock, exact.ExactMeritOracle,
        engine.TrajectoryOptions, engine_run, package,
        dict(kind="frozen_native_with_explicit_staged_engine", production_sha256=PRODUCTION_SHA256,
             engine_source_path=str(engine_source_path), engine_source_sha256=engine_source_sha256,
             source_pins=dict(SOURCE_PINS), shared_import_bootstrap_excluded_from_per_run_clock=True))


class _Timeline:
    def __init__(self, clock, started):
        if isinstance(started, (bool, np.bool_)):
            raise ValueError("a Boolean is not a true global clock start")
        self.clock, self.started, self.last = clock, float(started), float(started)
        if not math.isfinite(self.started):
            raise ValueError("finite true global start required")

    def absolute(self):
        raw = self.clock()
        now = float(raw)
        if isinstance(raw, (bool, np.bool_)) or not math.isfinite(now) or now < self.last:
            raise ValueError("clock must be finite and globally monotone")
        self.last = now
        return now

    def elapsed(self):
        return self.absolute() - self.started


class AnchorClock:
    """Passed ONLY to one engine; first call is its Ledger.started=t0."""
    def __init__(self, timeline):
        self.timeline, self.calls = timeline, 0

    def __call__(self):
        self.calls += 1
        return self.timeline.started if self.calls == 1 else self.timeline.absolute()


class PreparationContext:
    """Register partial executed-cost providers before preparation can fail."""
    def __init__(self):
        self.providers, self.records, self.identities = {}, {}, set()

    def register_preparation_costs(self, name, provider):
        identity = (id(getattr(provider, "__self__", provider)), id(getattr(provider, "__func__", provider)))
        if not isinstance(name, str) or not name or name in self.providers or identity in self.identities:
            raise ValueError("preparation counter provider/name must be unique")
        if not callable(provider):
            raise TypeError("cost provider must be callable")
        self.providers[name] = provider
        self.identities.add(identity)
        self.records[name] = dict(latest_counts=None, complete=False)

    def snapshot(self):
        for name, provider in self.providers.items():
            row = self.records[name]
            try:
                counts = _counts(provider())
                _monotone(row["latest_counts"] or {}, counts)
                row.update(latest_counts=counts, complete=True)
            except Exception as exc:
                row.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        return {k: dict(v) for k, v in self.records.items()}


class _OwnedDelegate:
    """No extra receipt/field/LU references: store authentic accepted headers."""
    def __init__(self, native):
        self.native, self.n = native, native.n
        self.contract = dict(native.contract)
        self.first_accepted = self.current_accepted = None
        self.counts = dict(evaluate_attempts=0, gradient_attempts=0, accept_attempts=0,
                           accepts_completed=0, cost_snapshot_calls=0)
        self.latest = None
        cold = _counts(native.cost_snapshot())
        if any(v for branch in cold.values() for v in branch.values()):
            raise ValueError("stage/metric owner was queried before registration")

    def evaluate(self, model):
        self.counts["evaluate_attempts"] += 1
        return self.native.evaluate(model)

    def gradient(self, receipt):
        self.counts["gradient_attempts"] += 1
        return self.native.gradient(receipt)

    def accept(self, receipt):
        self.counts["accept_attempts"] += 1
        self.native.accept(receipt)  # Public native owner/issued-receipt/fingerprint validation.
        self.counts["accepts_completed"] += 1
        header = dict(receipt.to_dict())
        if (header["geometry_sha256"] != self.contract["geometry_sha256"]
                or header["data_sha256"] != self.contract["data_sha256"]):
            raise ValueError("accepted receipt disagrees with owner contract")
        self.current_accepted = header
        if self.first_accepted is None:
            self.first_accepted = dict(header)

    def cost_snapshot(self):
        self.counts["cost_snapshot_calls"] += 1
        value = dict(native=_counts(self.native.cost_snapshot()), delegate=dict(self.counts))
        _monotone(self.latest or {}, value)
        self.latest = value
        return value

    def close(self):
        self.native = None


def _canonical_problem(prepared, api, context):
    if not isinstance(prepared, PreparedProblem) or type(prepared.physics) is not dict:
        raise TypeError("PreparedProblem.physics must be a configuration dict, never a preparation solver")
    config = dict(prepared.physics)
    if set(config) - {"shape", "spacing", "damping_width", "damping_strength", "factorization_cache_size"}:
        raise ValueError("unexpected native physics configuration key")
    shape = tuple(config["shape"])
    if len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in shape):
        raise ValueError("two positive integer native grid dimensions required")
    for key in ("spacing", "damping_strength"):
        v = config[key]
        if isinstance(v, (bool, np.bool_)) or not math.isfinite(v) or v < 0 or key == "spacing" and v == 0:
            raise ValueError("invalid native physics configuration: " + key)
    width = config["damping_width"]
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)) or width < 0:
        raise ValueError("invalid native damping width")
    cache = config.get("factorization_cache_size", 0)
    if isinstance(cache, (bool, np.bool_)) or cache != 0:
        raise ValueError("configuration templates must have disabled factor caches")
    config.update(shape=tuple(map(int, shape)), factorization_cache_size=0)
    m = np.asarray(prepared.initial_model)
    if (m.dtype != np.float64 or m.shape != (math.prod(shape),) or not np.isfinite(m).all()
            or (m < 1/36).any() or (m > 1).any()):
        raise ValueError("direct shared native-m initial bytes must satisfy[1/36,1]")
    receivers = tuple(prepared.receiver_indices)
    for index in receivers:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < m.size:
            raise ValueError("invalid receiver index")
    if not receivers:
        raise ValueError("nonempty ordered receivers required")
    blocks, source_order, seen, data_hash = [], {}, set(), hashlib.sha256()
    data_hash.update(json.dumps(list(map(int, receivers))).encode())
    for block in prepared.blocks:
        f, s, observed = float(block.frequency_hz), block.source_index, np.asarray(block.observed)
        if (isinstance(block.frequency_hz, (bool, np.bool_)) or f not in ALL8
                or isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer))):
            raise ValueError("fixed8 frequencies and integer sources required")
        if not 0 <= s < m.size or (s, f) in seen or observed.shape != (len(receivers),) or not np.isfinite(observed).all():
            raise ValueError("invalid/duplicate source-frequency observations")
        seen.add((s, f))
        source_order.setdefault(f, []).append(int(s))
        values = _freeze(observed, "<c16")
        label = str(block.label)
        data_hash.update(json.dumps([int(s), f, label], separators=(",", ":")).encode() + b"\0" + values.tobytes())
        blocks.append(api.block_type(int(s), f, values, label))
    if set(source_order) != set(ALL8) or any(source_order[f] != source_order[ALL8[0]] for f in ALL8):
        raise ValueError("all8 bands must use the same nonempty ordered source geometry")
    # No preparation solver/LU/field object is transferred into policy owners.
    # Real executed preparation counters are registered independently by prepare.
    return PreparedProblem(config, _freeze(m), tuple(blocks), tuple(map(int, receivers))), data_hash.hexdigest()


def _compact_engine(result):
    compact = {k: v for k, v in result.items() if k not in {"initial_model", "final_model"}}
    compact["checkpoints"] = [{k: v for k, v in c.items() if k != "model"} for c in result["checkpoints"]]
    compact["final_model_sha256"] = model_sha(result["final_model"])
    compact["initial_model_sha256"] = (model_sha(result["initial_model"]) if "initial_model" in result
        else result["checkpoints"][0]["model_sha256"] if result["checkpoints"] else None)
    return compact


def execute_timed(*, arm, prepare, api, settings, global_started, clock, attempt, progress=None):
    """Fully materialized paid policy API; actual return is stamped by caller.

    attempt is caller-owned and retains available prefixes on BaseException.
    It never contains truth or an oracle/receipt/field handle.
    """
    timeline, context = _Timeline(clock, global_started), PreparationContext()
    owners, owner_rows, identities = {}, [], set()
    pending = None
    initial = last_model = None
    common_initial = common_final = None
    valid_origin = None
    failed = False
    active_stage = None
    preparation_attempted = preparation_returned = False
    settings.validate()
    if arm not in ARMS:
        raise ValueError("unknown fixed arm")
    stages = [dict(index=i, frequencies_hz=list(freqs), status="unattempted",
        absolute_deadline_seconds=(settings.first_stage_deadline_seconds if len(ARMS[arm]) == 2 and i == 0 else settings.engine_deadline_seconds),
        maximum_steps=(settings.per_stage_maximum_steps if len(ARMS[arm]) == 2 else settings.single_stage_maximum_steps))
        for i, freqs in enumerate(ARMS[arm])]
    attempt.clear()
    attempt.update(arm=arm, status="running", phase="preparation", stages=stages,
        events=[], owners=owner_rows, common_initial_full8=None, common_final_full8=None,
        in_flight_work_unknown=False, source_binding=dict(api.source_binding))

    def event(phase, stage_index=None):
        attempt["phase"] = phase
        row = dict(phase=phase, stage_index=stage_index, elapsed_seconds=timeline.elapsed())
        attempt["events"].append(row)
        if progress is not None:
            progress(dict(arm=arm, **row))  # Bounded current event, never growing run history.

    def new_owner(label, role, frequencies, problem):
        row = dict(label=label, role=role, frequencies_hz=list(frequencies), status="construction_attempted",
                   latest_costs=None, costs_complete=False)
        owner_rows.append(row)  # Retain failed constructor denominator before calling it.
        try:
            template = api.physics_type(**dict(problem.physics))
            native = api.oracle_type(template,
                tuple(b for b in problem.blocks if float(b.frequency_hz) in frequencies),
                problem.receiver_indices, reference_package_root=api.package_root)
            if type(native) is not api.oracle_type or id(native) in identities:
                raise ValueError("new distinct native owner required")
            identities.add(id(native))
            delegate = _OwnedDelegate(native)
            owners[label] = delegate
            row.update(status="constructed_cold", contract=delegate.contract)
            return delegate
        except BaseException as exc:
            row.update(status="construction_failed", error=f"{type(exc).__name__}: {exc}",
                       in_flight_creation_costs_unknown=True)
            raise

    def paid_metric(owner, model):
        receipt = owner.evaluate(model)
        owner.accept(receipt)
        header = dict(owner.current_accepted)
        if header["model_sha256"] != model_sha(model) or not math.isfinite(header["objective"]) or header["objective"] < 0:
            raise ValueError("common objective receipt/model binding failed")
        return header

    try:
        event("preparation")
        preparation_attempted = True
        prepared = prepare(context)  # All loading, hashing and observation generation start after t0.
        preparation_returned = True
        problem, available_data_hash = _canonical_problem(prepared, api, context)
        initial = problem.initial_model
        attempt.update(initial_model_sha256=model_sha(initial), available8_data_sha256=available_data_hash,
            available_frequencies_hz=list(ALL8), bounds_m=[1/36, 1.], preparation_end_age_seconds=timeline.elapsed())
        if ARMS[arm][0] != ALL8 and timeline.elapsed() < stages[0]["absolute_deadline_seconds"]:
            event("common8_initial")
            metric = new_owner("common8_metric", "metric_only", ALL8, problem)
            common_initial = paid_metric(metric, initial)
            attempt["common_initial_full8"] = common_initial
        for stage in stages:
            index, deadline = stage["index"], stage["absolute_deadline_seconds"]
            if failed:
                stage["status"] = "unattempted_after_failure"
                continue
            if index and last_model is None:
                stage["status"] = "unattempted_no_prior_paid_state"
                continue
            if timeline.elapsed() >= deadline:
                stage["status"] = "unattempted_phase_deadline"
                continue
            stage_model = initial if index == 0 else last_model
            active_stage = stage
            stage.update(status="attempted", start_model_sha256=model_sha(stage_model),
                         call_start_age_seconds=timeline.elapsed())
            event("stage_start", index)
            owner = new_owner(f"stage{index+1}", "policy", tuple(stage["frequencies_hz"]), problem)
            anchor = AnchorClock(timeline)  # Do not consume outside the engine Ledger.
            options = api.options_type(budget_seconds=deadline, maximum_steps=stage["maximum_steps"],
                initial_step=.002, maximum_step=.02, maximum_line_search_trials=settings.maximum_line_search_trials,
                contraction=.5, accepted_step_growth=1.5, relative_objective_target=None, absolute_objective_target=None)
            result = api.engine_run(oracle_factory=lambda owner=owner: owner, m0=stage_model,
                lower_bound=1/36, upper_bound=1., options=options, clock=anchor,
                maximum_corrections=settings.maximum_corrections, function_tolerance=settings.function_tolerance,
                gradient_tolerance=settings.gradient_tolerance)
            stage.update(engine_result=_compact_engine(result), anchor_clock_calls=anchor.calls,
                         call_return_age_seconds=timeline.elapsed(), status=result["status"])
            callbacks = [s["native_callback"] for s in result["steps"] if "native_callback" in s]
            stage["native_callback_events"] = len(callbacks)
            stage["callback_count_scope"] = "recorded native_callback events, including equal or ineligible events"
            if len(callbacks) > stage["maximum_steps"]:
                raise ValueError("native callback safety cap exceeded")
            engine_failed = result["status"] == "failed" or not result.get("costs_complete")
            stage["method_failed"] = bool(engine_failed)
            eligible = [c for c in result["checkpoints"] if c.get("within_budget") is True
                and not isinstance(c.get("elapsed_seconds"), bool)
                and math.isfinite(c["elapsed_seconds"]) and 0 <= c["elapsed_seconds"] <= deadline]
            stage["eligible_checkpoint_count"] = len(eligible)
            if not eligible:
                stage["paid_state_available"] = False
                failed = failed or engine_failed
                active_stage = None
                continue  # In stage2 retain only the PREVIOUS chronological paid model.
            first = eligible[0]
            if (first["kind"] != "initial" or first["model_sha256"] != stage["start_model_sha256"]
                    or owner.first_accepted is None or owner.first_accepted["model_sha256"] != first["model_sha256"]
                    or owner.first_accepted["objective"] != first["objective"]):
                raise ValueError("stage initial checkpoint/authentic receipt binding failed")
            if any(a["elapsed_seconds"] > b["elapsed_seconds"] for a, b in zip(eligible, eligible[1:], strict=False)):
                raise ValueError("eligible checkpoints are not chronological")
            final = _freeze(result["final_model"])
            if eligible[-1]["model_sha256"] != model_sha(final):
                raise ValueError("engine final model is not its last eligible checkpoint")
            if (owner.current_accepted is None or owner.current_accepted["model_sha256"] != model_sha(final)
                    or owner.current_accepted["objective"] != eligible[-1]["objective"]
                    or result["final_objective"] != eligible[-1]["objective"]):
                raise ValueError("last eligible checkpoint is not the owner's authentic current receipt")
            if not np.isfinite(final).all() or (final < 1/36).any() or (final > 1).any():
                raise ValueError("returned native model violates shared bounds")
            last_model = final
            valid_origin = dict(stage_index=index, model_sha256=model_sha(final),
                                elapsed_seconds=eligible[-1]["elapsed_seconds"])
            stage.update(paid_state_available=True, return_model_origin=valid_origin)
            if index == 0 and tuple(stage["frequencies_hz"]) == ALL8:
                common_initial = dict(owner.first_accepted)
                attempt["common_initial_full8"] = common_initial
            failed = failed or engine_failed
            result = None  # Retain scalar/hash/counter events, not dense checkpoint histories.
            active_stage = None
            event("stage_return", index)
        if not failed and last_model is not None and common_initial is not None and timeline.elapsed() < settings.api_return_cap_seconds:
            event("common8_final")
            full_owners = [owners[r["label"]] for r in owner_rows
                if r["label"] in owners and r["frequencies_hz"] == list(ALL8)]
            if not full_owners:
                raise ValueError("no legitimate owned full8 metric oracle")
            common_final = paid_metric(full_owners[-1], last_model)
            if any(common_final[k] != common_initial[k] for k in ("geometry_sha256", "data_sha256")):
                raise ValueError("common initial/final full8 survey or observations differ")
            attempt["common_final_full8"] = common_final
        attempt.update(status="failed_method" if failed else "completed_paid_state" if last_model is not None and common_final is not None else "incomplete_no_paid_state_or_metric",
                       final_model_origin=valid_origin)
        event("finalizing_payload")
    except Exception as exc:
        failed = True
        attempt.update(status="failed_controller", error=f"{type(exc).__name__}: {exc}")
        if active_stage is not None:
            active_stage.update(status="failed_controller", error=attempt["error"])
    except BaseException as exc:
        failed = True
        pending = exc
        attempt.update(status="interrupted", error=f"{type(exc).__name__}: {exc}", in_flight_work_unknown=True)
        if active_stage is not None:
            active_stage.update(status="interrupted", error=attempt["error"])
        prefix = getattr(exc, "bounded_exact_prefix", None)
        if prefix is not None and active_stage is not None:
            try:
                active_stage["engine_result"] = _compact_engine(prefix)
                checkpoints = [c for c in prefix["checkpoints"] if c.get("within_budget") is True
                    and math.isfinite(c["elapsed_seconds"]) and 0 <= c["elapsed_seconds"] <= active_stage["absolute_deadline_seconds"]]
                if checkpoints and checkpoints[-1]["model_sha256"] == model_sha(prefix["final_model"]):
                    last_model = _freeze(prefix["final_model"])
                    valid_origin = dict(stage_index=active_stage["index"], model_sha256=model_sha(last_model),
                        elapsed_seconds=checkpoints[-1]["elapsed_seconds"])
                    attempt["final_model_origin"] = valid_origin
            except Exception as prefix_error:
                active_stage["prefix_retention_error"] = f"{type(prefix_error).__name__}: {prefix_error}"
    finally:
        for row in owner_rows:
            owner = owners.get(row["label"])
            if owner is not None:
                try:
                    row.update(latest_costs=owner.cost_snapshot(), costs_complete=True)
                except Exception as exc:
                    row.update(latest_costs=owner.latest, costs_complete=False, cost_error=f"{type(exc).__name__}: {exc}")
        prep_records = context.snapshot()
        attempt["preparation_costs"] = prep_records
        attempt["preparation_in_flight_counts_unknown"] = preparation_attempted and not preparation_returned
        attempt["costs_complete"] = (preparation_returned and not attempt["in_flight_work_unknown"]
            and bool(prep_records) and all(r["complete"] for r in prep_records.values())
            and all(r["costs_complete"] for r in owner_rows))
        try:
            owner_sum = _sum_counts([r["latest_costs"] for r in owner_rows if r["latest_costs"] is not None])
            prep_sum = _sum_counts([r["latest_counts"] for r in prep_records.values() if r["latest_counts"] is not None])
            owners_complete = not attempt["in_flight_work_unknown"] and all(r["costs_complete"] for r in owner_rows)
            prep_complete = preparation_returned and bool(prep_records) and all(r["complete"] for r in prep_records.values())
            attempt["owner_costs_summed_once"] = owner_sum if owners_complete else None
            attempt["preparation_costs_separate"] = prep_sum if prep_complete else None
            attempt["owner_known_counts_partial_sum"] = None if owners_complete else owner_sum
            attempt["preparation_known_counts_partial_sum"] = None if prep_complete else prep_sum
        except Exception as exc:
            attempt.update(costs_complete=False, cost_aggregation_error=f"{type(exc).__name__}: {exc}")
        for owned in owners.values():
            owned.close()  # Release owned LU/field references before caller-observed API return.
        context.providers.clear()
    payload = dict(attempt)
    payload.update(initial_model=initial, final_model=last_model, settings=asdict(settings),
        final_model_sha256=None if last_model is None else model_sha(last_model),
        truth_input=False, checkpoint_vectors_retained=False,
        elapsed_timestamps_are_global_ages_not_additive_durations=True,
        last_observed_absolute_clock=timeline.last,
        runtime_eligibility_requires_caller_return_stamp=True)
    a = None if common_initial is None else common_initial["objective"]
    b = None if common_final is None else common_final["objective"]
    ratio = b/a if a is not None and b is not None and a > 0 else None
    payload["common_full8_ratio"] = ratio if ratio is not None and math.isfinite(ratio) else None
    payload["common_initial_objective_zero"] = a == 0 if a is not None else None
    payload["common_ratio_unavailable_reason"] = ("missing_paid_full8_objective" if a is None or b is None
        else "zero_initial_objective" if a == 0 else "nonrepresentable_ratio" if not math.isfinite(ratio) else None)
    payload["method_valid_before_return_gate"] = not failed and last_model is not None and common_initial is not None and common_final is not None and attempt["costs_complete"]
    json.dumps(_jsonable(payload), allow_nan=False)  # Included in measured API tail.
    attempt["payload"] = payload
    if pending is not None:
        raise pending
    return payload


def measure_controller_return(*, arm, prepare, api, settings=None, clock=perf_counter, attempt=None, progress=None):
    """Measurement harness: stamp execute_timed's actual return, then classify."""
    settings = ControllerSettings() if settings is None else settings
    attempt = {} if attempt is None else attempt
    started = clock()  # True global start, before preparation or per-run validation.
    payload = execute_timed(arm=arm, prepare=prepare, api=api, settings=settings,
        global_started=started, clock=clock, attempt=attempt, progress=progress)
    returned = clock()  # All paid API work and strict finish metadata have returned.
    wall = float(returned) - float(started)
    valid_time = (not isinstance(returned, (bool, np.bool_)) and math.isfinite(wall)
        and float(returned) >= payload["last_observed_absolute_clock"]
        and 0 <= wall <= settings.api_return_cap_seconds)
    eligible = bool(payload["method_valid_before_return_gate"] and valid_time)
    return dict(payload=payload,
        timing=dict(global_started=started, actual_api_returned=returned, actual_api_return_seconds=wall,
                    api_return_cap_seconds=settings.api_return_cap_seconds,
                    return_stamp_monotone=bool(float(returned) >= payload["last_observed_absolute_clock"]),
                    actual_api_overshoot_seconds=max(0., wall-settings.api_return_cap_seconds),
                    scope="execute_timed full return; measurement classification/artifact persistence afterward",
                    preparation_and_stage_ages_not_added_again=True),
        runtime_gate=dict(eligible=eligible, reason="fully_returned_paid_state" if eligible else "failed_incomplete_or_late",
                          event_seconds=wall if eligible else None))
