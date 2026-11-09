"""Held classical residual/Jacobian callbacks; no data loader or study driver.

Every field point owns copies and retains its genuine native receipt. Old
LinearOperators remain fixed after later trials. No native factor is accessed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import weakref
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator

_LIVE_NATIVE_OWNERS = weakref.WeakKeyDictionary()
_LIVE_MAP_BUILDERS = weakref.WeakKeyDictionary()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def _finite_positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _indices(values, n, name, *, unique=False):
    result = tuple(values)
    if (not result or any(isinstance(x, (bool, np.bool_))
            or not isinstance(x, (int, np.integer)) or not 0 <= x < n for x in result)):
        raise ValueError(f"{name} must contain in-range integer positions")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return tuple(map(int, result))


@dataclass(frozen=True)
class Layout:
    """Declared full objective order, independent of exported frame order."""
    n: int
    receivers: tuple[int, ...]
    # frequency, source indices in column order, global observation block indices
    frames: tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError("positive integer n required")
        receivers = _indices(self.receivers, self.n, "receivers")
        frames, frequencies, blocks = [], set(), []
        for frequency, sources, indices in self.frames:
            frequency = _finite_positive(frequency, "frequency")
            if frequency in frequencies or not math.isfinite((2*math.pi*frequency)**2):
                raise ValueError("distinct representable frequencies required")
            frequencies.add(frequency)
            sources = _indices(sources, self.n, "sources", unique=True)
            indices = tuple(indices)
            if (len(indices) != len(sources) or any(isinstance(i, (bool, np.bool_))
                    or not isinstance(i, (int, np.integer)) or i < 0 for i in indices)):
                raise ValueError("one nonnegative block index per source required")
            indices = tuple(map(int, indices))
            blocks.extend(indices)
            frames.append((frequency, sources, indices))
        if not frames or sorted(blocks) != list(range(len(blocks))):
            raise ValueError("all observation blocks must occur exactly once")
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "frames", tuple(frames))

    @property
    def blocks(self):
        return sum(len(sources) for _, sources, _ in self.frames)

    @property
    def complex_terms(self):
        return self.blocks*len(self.receivers)


@dataclass(frozen=True)
class CopiedFrame:
    frequency: float
    sources: tuple[int, ...]
    blocks: tuple[int, ...]
    unique: tuple[int, ...]
    expand: tuple[int, ...]
    amplitudes: tuple[complex, ...]
    symmetric: bool
    state: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    observed: np.ndarray = field(repr=False)


@dataclass(frozen=True, eq=False)
class FieldPoint:
    identity: tuple
    model: np.ndarray = field(repr=False)
    config: tuple
    frames: tuple[CopiedFrame, ...]
    objective: float
    native_receipt: object = field(repr=False)
    native_snapshot: object = field(repr=False)
    token: object = field(repr=False)
    fingerprint: str


class CachedJacobianCallbacks:
    """Source-level adapter, not a bounded optimizer or execution permission.

    The native delegate must implement evaluate/export_merit_fields/
    validate_merit_fields/cost_snapshot. A public map builder supplies only
    missing rows. Native integration must bind the actual frozen classes.
    """

    def __init__(self, native, builder, *, initial_model, layout, clock=time.perf_counter):
        self.native, self.builder, self.layout, self.clock = native, builder, layout, clock
        self.started = self.clock()
        previous = _LIVE_NATIVE_OWNERS.get(native)
        if previous is not None and previous() is not None:
            raise ValueError("native owner already belongs to another live callback policy")
        previous = _LIVE_MAP_BUILDERS.get(builder)
        if previous is not None and previous() is not None:
            raise ValueError("map builder already belongs to another live callback policy")
        builder_prefix = builder.cost_snapshot()
        if builder_prefix["cold_owner_batches"] or builder_prefix["unknown_work_present"]:
            raise ValueError("map builder must start with no prior numerical work")
        self.counts = {}
        self.token = object()
        self.latest_fun = self.latest_jac = None
        self.last_operator = self.last_operator_identity = None
        self.last_failure = None
        self.phi0 = self.geometry = self.data = None
        self.native_cost = {"last_known_cumulative": None, "unknown_since_prefix": True}
        self.native_failed_dispatch_work_unknown = False
        self._capture_native()
        if self.native_cost["unknown_since_prefix"]:
            raise ValueError("a known cold native cost snapshot is required")
        self._require_cold(self.native_cost["last_known_cumulative"])
        self.initial_model = self._freeze(self._model(initial_model), np.float64)
        _LIVE_NATIVE_OWNERS[native] = weakref.ref(self)
        _LIVE_MAP_BUILDERS[builder] = weakref.ref(self)

    def _count(self, name, amount=1):
        self.counts[name] = self.counts.get(name, 0)+int(amount)

    @staticmethod
    def _require_cold(value):
        if isinstance(value, dict):
            for child in value.values():
                CachedJacobianCallbacks._require_cold(child)
        elif isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError("native owner must start with known zero executed work")

    def _capture_native(self):
        self._count("native_cost_snapshot_attempts")
        try:
            value = copy.deepcopy(self.native.cost_snapshot())
            json.dumps(value, allow_nan=False)
            self.native_cost = {"last_known_cumulative": value, "unknown_since_prefix": False}
            if self.native_failed_dispatch_work_unknown:
                self.native_cost.update(unknown_since_prefix=True, failed_dispatch_work_may_exceed_prefix=True)
            self._count("native_cost_snapshots_completed")
        except BaseException as exc:  # noqa: BLE001 -- preserve interrupts without replacing original failure
            self.native_cost["unknown_since_prefix"] = True
            self.native_cost["snapshot_failure_type"] = type(exc).__name__

    def _native(self, name, *args):
        self._count(f"native_{name}_attempts")
        self.native_cost["unknown_since_prefix"] = True
        completed = False
        try:
            result = getattr(self.native, name)(*args)
            self._count(f"native_{name}_completed")
            completed = True
            return result
        finally:
            self._capture_native()
            if not completed:
                self.native_failed_dispatch_work_unknown = True
                self.native_cost["unknown_since_prefix"] = True
                self.native_cost["failed_dispatch_work_may_exceed_prefix"] = True

    def _model(self, model):
        value = np.asarray(model)
        if (value.dtype != np.float64 or value.shape != (self.layout.n,)
                or not np.isfinite(value).all() or np.any(value <= 0)):
            raise ValueError("positive finite flat native float64 m required")
        return value

    def _freeze(self, values, dtype):
        values = np.asarray(values)
        if values.dtype != np.dtype(dtype) or not np.isfinite(values).all():
            raise ValueError("finite exact-dtype array required")
        self._count("explicit_copy_bytes", values.nbytes)
        self._count("explicit_array_copies")
        return np.frombuffer(values.tobytes(order="C"), dtype=dtype).reshape(values.shape)

    def _hash(self, value):
        self._count("fingerprint_array_bytes", np.asarray(value).nbytes)
        self._count("explicit_copy_bytes", np.asarray(value).nbytes)
        self._count("fingerprint_byte_copies")
        return array_sha(value)

    def _fingerprint(self, model, frames, identity, config):
        terms = [identity, config, self._hash(model)]
        for frame in frames:
            terms.append((frame.frequency, frame.sources, frame.blocks, frame.unique,
                frame.expand, tuple((a.real, a.imag) for a in frame.amplitudes), frame.symmetric,
                self._hash(frame.state), self._hash(frame.residual), self._hash(frame.observed)))
        return hashlib.sha256(repr(terms).encode()).hexdigest()

    def _copy_point(self, receipt, snapshot, requested):
        model = self._freeze(self._model(snapshot.model), np.float64)
        if not np.array_equal(model, requested) or not np.array_equal(model, receipt.model):
            raise ValueError("requested/native/exported model mismatch")
        if snapshot.model_sha256 != self._hash(model):
            raise ValueError("exported model identity mismatch")
        identity = (snapshot.receipt_key, snapshot.model_sha256, snapshot.operator_sha256,
                    snapshot.geometry_sha256, snapshot.data_sha256)
        if (receipt.key, receipt.model_sha256, receipt.operator_sha256,
                receipt.geometry_sha256, receipt.data_sha256) != identity:
            raise ValueError("receipt/export provenance mismatch")
        if self.geometry is not None and (identity[3], identity[4]) != (self.geometry, self.data):
            raise ValueError("survey/data changed")
        if (isinstance(snapshot.damping_width, (bool, np.bool_))
                or not isinstance(snapshot.damping_width, (int, np.integer))
                or isinstance(snapshot.spacing, (bool, np.bool_))
                or isinstance(snapshot.damping_strength, (bool, np.bool_))):
            raise TypeError("physical configuration scalar types changed")
        config = (tuple(snapshot.shape), float(snapshot.spacing), int(snapshot.damping_width),
                  float(snapshot.damping_strength))
        if (math.prod(config[0]) != self.layout.n or len(config[0]) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in config[0])
                or not math.isfinite(config[1]) or config[1] <= 0 or config[2] < 0
                or not math.isfinite(config[3]) or config[3] < 0):
            raise ValueError("invalid physical configuration")
        expected = {frequency: (sources, blocks) for frequency, sources, blocks in self.layout.frames}
        frames, seen, phi = [], set(), 0.0
        for source in snapshot.frequencies:
            frequency = float(source.frequency_hz)
            if frequency in seen or frequency not in expected:
                raise ValueError("missing/extra/duplicate frequency")
            seen.add(frequency)
            sources, blocks = expected[frequency]
            if tuple(source.source_indices) != sources or tuple(source.block_indices) != blocks:
                raise ValueError("source columns/observation blocks changed")
            unique = tuple(dict.fromkeys(self.layout.receivers))
            expand = tuple(unique.index(i) for i in self.layout.receivers)
            if (tuple(source.receiver_indices) != self.layout.receivers
                    or tuple(source.unique_receiver_indices) != unique
                    or tuple(source.receiver_expand_indices) != expand):
                raise ValueError("ordered receiver slots changed")
            if source.cache_kind != "forward_states":
                raise ValueError("held component requires paid U; receiver-map route needs separate integration")
            amplitudes = np.asarray(source.source_amplitude, dtype=np.complex128)
            if amplitudes.ndim == 0:
                amplitudes = np.full(len(sources), complex(amplitudes), dtype=np.complex128)
                self._count("temporary_array_bytes", amplitudes.nbytes)
            if (amplitudes.shape != (len(sources),) or not np.isfinite(amplitudes).all()
                    or np.any(amplitudes == 0)):
                raise ValueError("finite nonzero point-source amplitude per column required")
            state = self._freeze(source.cached_field, np.complex128)
            residual = self._freeze(source.residual_slots, np.complex128)
            observed = self._freeze(source.observed_slots, np.complex128)
            if (state.shape != (self.layout.n, len(sources))
                    or residual.shape != (len(self.layout.receivers), len(sources))
                    or observed.shape != residual.shape):
                raise ValueError("field/residual dimensions changed")
            predicted_residual = state[list(self.layout.receivers)]-observed
            self._count("temporary_array_bytes", 2*residual.nbytes)
            self._count("residual_consistency_complex_terms", residual.size)
            if not np.allclose(predicted_residual, residual, rtol=2e-12, atol=2e-14):
                raise ValueError("paid U/residual/observation inconsistency")
            phi += .5*float(np.vdot(residual, residual).real)
            self._count("objective_complex_terms", residual.size)
            if not isinstance(source.transpose_symmetric, (bool, np.bool_)):
                raise TypeError("actual Boolean transpose-symmetry result required")
            frames.append(CopiedFrame(frequency, sources, blocks, unique, expand,
                tuple(map(complex, amplitudes)), bool(source.transpose_symmetric), state, residual, observed))
        if seen != set(expected):
            raise ValueError("all declared frequencies required")
        objective = float(receipt.objective)
        if not math.isfinite(objective) or objective < 0 or not math.isclose(phi, objective, rel_tol=2e-12, abs_tol=2e-14):
            raise ValueError("native Phi differs from full residual objective")
        frames = tuple(frames)
        fingerprint = self._fingerprint(model, frames, identity, config)
        return FieldPoint(identity, model, config, frames, objective, receipt, snapshot, self.token, fingerprint)

    def _validate(self, point):
        self._count("point_validations")
        if (type(point) is not FieldPoint or point.token is not self.token
                or point.fingerprint != self._fingerprint(point.model, point.frames, point.identity, point.config)):
            raise ValueError("foreign or altered copied field point")
        self._native("validate_merit_fields", point.native_snapshot)

    def _failure(self, phase, exc):
        try:
            builder_prefix = self.builder.cost_snapshot()
        except BaseException as snapshot_exc:  # noqa: BLE001 -- diagnostic failure cannot mask original interrupt
            builder_prefix = {"unknown_work_present": True, "snapshot_failure_type": type(snapshot_exc).__name__}
        self.last_failure = {"phase": phase, "exception_type": type(exc).__name__,
            "native_cost": copy.deepcopy(self.native_cost), "counts": dict(self.counts),
            "map_builder_prefix": builder_prefix}

    def fun(self, model):
        self._count("residual_requests")
        try:
            model = self._model(model)
            if self.phi0 is None and not np.array_equal(model, self.initial_model):
                raise ValueError("first query must be registered direct initial m")
            receipt = self._native("evaluate", model)
            snapshot = self._native("export_merit_fields", receipt)
            self._native("validate_merit_fields", snapshot)
            point = self._copy_point(receipt, snapshot, model)
            if self.phi0 is None:
                self.phi0 = _finite_positive(point.objective, "Phi0")
                self.geometry, self.data = point.identity[3:]
            result = np.empty((self.layout.blocks, len(self.layout.receivers)), dtype=np.complex128)
            self._count("allocated_array_bytes", result.nbytes)
            for frame in point.frames:
                result[list(frame.blocks)] = frame.residual.T
                self._count("residual_scatter_complex_terms", frame.residual.size)
            value = self._real_stack(result)
            self.latest_fun = point
            self._count("residual_requests_completed")
            return value
        except BaseException as exc:
            self._failure("residual", exc)
            raise

    def _real_stack(self, values):
        result = np.empty(2*self.layout.complex_terms, dtype=np.float64)
        self._count("allocated_array_bytes", result.nbytes)
        result[:self.layout.complex_terms] = values.real.reshape(-1)
        result[self.layout.complex_terms:] = values.imag.reshape(-1)
        result /= math.sqrt(self.phi0)
        self._count("real_terms_emitted_and_scaled", result.size)
        return result

    def jac(self, model):
        self._count("jacobian_requests")
        try:
            model = self._model(model)
            point = next((p for p in (self.latest_fun, self.latest_jac)
                if p is not None and np.array_equal(p.model, model)), None)
            if point is None:
                raise ValueError("jac requires a completed residual point; no hidden merit query")
            self._validate(point)
            if point.identity == self.last_operator_identity:
                self._count("jacobian_operator_cache_hits")
                self._count("jacobian_requests_completed")
                return self.last_operator
            maps = []
            for frame in point.frames:
                source_positions = {source: i for i, source in enumerate(frame.sources)}
                missing = [receiver for receiver in frame.unique
                    if not frame.symmetric or receiver not in source_positions]
                rows = np.empty((len(frame.unique), self.layout.n), dtype=np.complex128)
                self._count("allocated_array_bytes", rows.nbytes)
                for j, receiver in enumerate(frame.unique):
                    if receiver not in missing:
                        column = source_positions[receiver]
                        rows[j] = frame.state[:, column]/frame.amplitudes[column]
                        self._count("reciprocity_complex_divisions", self.layout.n)
                        self._count("temporary_array_bytes", self.layout.n*16)
                if missing:
                    self._count("map_build_attempts")
                    supplied = self.builder.build(point=point, frequency=frame.frequency, receivers=tuple(missing))
                    if (np.asarray(supplied).dtype != np.complex128 or np.asarray(supplied).shape != (len(missing), self.layout.n)
                            or not np.isfinite(supplied).all()):
                        raise ValueError("missing-row builder returned invalid rows")
                    for j, receiver in enumerate(missing):
                        rows[frame.unique.index(receiver)] = supplied[j]
                    self._count("map_builds_completed")
                    self._count("map_scatter_complex_terms", np.asarray(supplied).size)
                maps.append(self._freeze(rows, np.complex128))
            maps = tuple(maps)
            map_hashes = tuple(self._hash(value) for value in maps)

            def checked():
                self._validate(point)
                if tuple(self._hash(value) for value in maps) != map_hashes:
                    raise ValueError("copied receiver map altered")

            def matvec(vector):
                self._count("jvp_attempts")
                try:
                    checked()
                    vector = self._real_vector(vector, self.layout.n)
                    data = np.empty((self.layout.blocks, len(self.layout.receivers)), dtype=np.complex128)
                    self._count("allocated_array_bytes", data.nbytes)
                    for frame, rows in zip(point.frames, maps, strict=True):
                        weighted = frame.state*vector[:, None]
                        contracted = rows@weighted
                        expanded = contracted[list(frame.expand)]*((2*math.pi*frame.frequency)**2)
                        self._count("temporary_array_bytes", weighted.nbytes+contracted.nbytes+2*expanded.nbytes)
                        self._count("jvp_elementwise_complex_products", weighted.size+expanded.size)
                        self._count("jvp_dense_dot_complex_products", rows.shape[0]*rows.shape[1]*weighted.shape[1])
                        data[list(frame.blocks)] = expanded.T
                    result = self._real_stack(data)
                    self._count("jvp_completed")
                    return result
                except BaseException as exc:
                    self._failure("jvp", exc)
                    raise

            def rmatvec(vector):
                self._count("vjp_attempts")
                try:
                    checked()
                    vector = self._real_vector(vector, 2*self.layout.complex_terms)
                    dual = (vector[:self.layout.complex_terms]+1j*vector[self.layout.complex_terms:])/math.sqrt(self.phi0)
                    dual = dual.reshape(self.layout.blocks, len(self.layout.receivers))
                    result = np.zeros(self.layout.n, dtype=np.float64)
                    self._count("allocated_array_bytes", result.nbytes)
                    self._count("temporary_array_bytes", 3*dual.nbytes)
                    for frame, rows in zip(point.frames, maps, strict=True):
                        slots = dual[list(frame.blocks)].T
                        unique = np.zeros((len(frame.unique), len(frame.sources)), dtype=np.complex128)
                        np.add.at(unique, np.asarray(frame.expand), slots)
                        conjugated = np.conjugate(unique)
                        coded = rows.T@conjugated
                        products = frame.state*coded
                        summed = np.sum(products, axis=1)
                        result += ((2*math.pi*frame.frequency)**2)*summed.real
                        self._count("temporary_array_bytes", slots.nbytes+unique.nbytes+conjugated.nbytes+coded.nbytes+products.nbytes+summed.nbytes+result.nbytes)
                        self._count("vjp_slot_accumulations", slots.size)
                        self._count("vjp_dense_dot_complex_products", rows.shape[0]*rows.shape[1]*unique.shape[1])
                        self._count("vjp_elementwise_complex_products", products.size)
                        self._count("vjp_real_accumulations", self.layout.n)
                    self._count("vjp_completed")
                    return result
                except BaseException as exc:
                    self._failure("vjp", exc)
                    raise

            operator = LinearOperator((2*self.layout.complex_terms, self.layout.n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
            self.latest_jac = point
            self.last_operator_identity, self.last_operator = point.identity, operator
            self._count("jacobian_requests_completed")
            return operator
        except BaseException as exc:
            self._failure("jacobian", exc)
            raise

    @staticmethod
    def _real_vector(value, n):
        value = np.asarray(value)
        if (value.dtype != np.float64 or value.shape not in ((n,), (n, 1))
                or not np.isfinite(value).all()):
            raise ValueError("finite real float64 direction/dual required")
        return value.reshape(n)

    def cost_snapshot(self):
        """No native re-query: latest observed cumulative prefixes, not their sum."""
        return {"component_counts": dict(self.counts), "native": copy.deepcopy(self.native_cost),
            "map_builder": self.builder.cost_snapshot(), "elapsed_seconds": self.clock()-self.started,
            "copy_scope": "All explicit component field copies counted. Known owned array allocations/temporary bytes and logical algebra terms counted; NumPy/SciPy/native internal allocations and FLOPs are not independently instrumented, and remain timed/unknown rather than asserted zero.",
            "cost_scope": "Native cumulative snapshot counted ONCE. Each map batch owns a separate cold physics instance; its prefixes are disjoint from native work. Snapshot/query dispatch counts are not RHS equivalents."}


if __name__ == "__main__":
    print(json.dumps({"status": "HELD_COMPONENT_ONLY", "project_execution_available": False,
        "scope": "Classical cached-J residual callbacks; no project driver, data loader, prior or student design."}))
