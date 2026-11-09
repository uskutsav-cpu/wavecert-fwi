"""Standard physical-velocity coordinates around the paid native merit oracle.

No optimizer, dataset, policy, step schedule or physics approximation is added.
All optimizer-facing models are velocity; all native physics models are m=v^-2.
"""
from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np
from wavecert.physics.exact_merit import ExactMeritOracle

NATIVE_SOURCE_SHA256 = "0fec1f215b28e36017c1e28409d7f7e909a03cbf5b717de5d23eb8d52eb26810"
_CLAIMED_NATIVES = weakref.WeakSet()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _immutable(values):
    array = np.ascontiguousarray(values, dtype="<f8")
    return np.frombuffer(array.tobytes(), dtype="<f8").reshape(array.shape)


def model_sha256(values):
    return _sha(np.ascontiguousarray(values, dtype="<f8").tobytes())


def _vector(values, *, size=None, positive=True):
    raw = np.asarray(values)
    if (raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf"
            or (size is not None and raw.size != size)):
        raise ValueError("nonempty flat real numeric vector of the correct size required")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all() or (positive and (array <= 0).any()):
        raise ValueError("finite positive model required" if positive else "finite direction required")
    return array


def velocity_to_slowness(velocity):
    """Actual canonical FP mapping, not a promise of inverse roundtrip identity."""
    velocity = _vector(velocity)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        model = np.power(velocity, -2.)
    if not np.isfinite(model).all() or (model <= 0).any():
        raise ValueError("inverse-square model is not representable as positive finite float64")
    return _immutable(model)


def slowness_to_velocity(model):
    model = _vector(model)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        velocity = 1. / np.sqrt(model)
    if not np.isfinite(velocity).all() or (velocity <= 0).any():
        raise ValueError("velocity is not representable as positive finite float64")
    return _immutable(velocity)


def start_from_slowness(requested_model):
    """Return v, its actual native m, and both initial hashes without correction."""
    requested = _immutable(_vector(requested_model))
    velocity = slowness_to_velocity(requested)
    actual = velocity_to_slowness(velocity)
    return velocity, actual, dict(requested_native_model_sha256=model_sha256(requested),
        velocity_model_sha256=model_sha256(velocity), actual_native_model_sha256=model_sha256(actual),
        native_roundtrip_bitwise_equal=bool(np.array_equal(requested, actual)),
        native_roundtrip_max_abs_difference=float(np.max(np.abs(requested - actual))),
        convention="v=1/sqrt(requested_m), actual_m=np.power(v,-2); no ULP search, replacement or cached-receipt borrowing")


def _bounds(lower, upper, size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("positive integer bound-vector size required")
    for value in [lower, upper]:
        raw = np.asarray(value)
        if raw.ndim > 1 or raw.dtype.kind not in "iuf":
            raise ValueError("bounds must be real numeric scalars or flat vectors")
    lo, hi = (np.array(np.broadcast_to(value, (size,)), dtype=np.float64) for value in [lower, upper])
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo <= 0).any() or (lo > hi).any():
        raise ValueError("finite positive ordered bounds required")
    return lo, hi


def velocity_bounds_to_slowness(lower_velocity, upper_velocity, *, size):
    lo, hi = _bounds(lower_velocity, upper_velocity, size)
    return velocity_to_slowness(hi), velocity_to_slowness(lo)


def slowness_bounds_to_velocity(lower_model, upper_model, *, size):
    lo, hi = _bounds(lower_model, upper_model, size)
    return slowness_to_velocity(hi), slowness_to_velocity(lo)


def _native_fingerprint(receipt):
    return tuple(getattr(receipt, key) for key in ["objective", "key", "model_sha256",
                                                  "operator_sha256", "geometry_sha256", "data_sha256"])


@dataclass(frozen=True, eq=False)
class VelocityMeritReceipt:
    objective: float
    key: str
    model_sha256: str
    model: np.ndarray = dc_field(repr=False)
    problem_fingerprint_sha256: str
    _native: object = dc_field(repr=False)
    _owner: object = dc_field(repr=False)

    @property
    def native_model(self):
        return self._native.model

    @property
    def native_model_sha256(self):
        return self._native.model_sha256

    def to_dict(self):
        return dict(parameterization="physical_velocity", objective=self.objective, key=self.key,
            velocity_model_sha256=self.model_sha256, actual_native_model_sha256=self.native_model_sha256,
            problem_fingerprint_sha256=self.problem_fingerprint_sha256, native_receipt=self._native.to_dict())


@dataclass(frozen=True)
class VelocityGradientEvaluation:
    objective: float
    gradient: np.ndarray
    receipt: VelocityMeritReceipt
    additional_rhs: int
    frequency_strategies: dict


class VelocityMeritOracle:
    """One cold native oracle per policy; retain native public validation/cache work.

    Instantiate inside the existing engine's oracle_factory. Units are an
    explicit label for caller-consistent native length/time units, never a
    silent conversion. No certificate-field export or tangent solve is added.
    """

    def __init__(self, native, *, velocity_units):
        if type(native) is not ExactMeritOracle:
            raise TypeError("fresh frozen ExactMeritOracle required")
        if not isinstance(velocity_units, str) or not velocity_units.strip():
            raise ValueError("explicit nonempty native-consistent velocity units required")
        if native in _CLAIMED_NATIVES or any(value for group in native.cost_snapshot().values() for value in group.values()):
            raise ValueError("fresh unqueried native oracle required for each velocity policy")
        self._source_path = Path(__file__).resolve()
        self._native_path = Path(ExactMeritOracle.evaluate.__code__.co_filename).resolve()
        self._source_sha = _sha(self._source_path.read_bytes())
        if _sha(self._native_path.read_bytes()) != NATIVE_SOURCE_SHA256:
            raise RuntimeError("frozen native exact-merit source changed")
        self._native, self._token = native, object()
        self._issued = weakref.WeakKeyDictionary()
        self._current = self._trial = self._initial = None
        self._cost = dict(constructions=1)
        parameterization = dict(name="physical_velocity", velocity_units=velocity_units,
            native_squared_slowness_units="(" + velocity_units + ")^-2", mapping="np.power(float64_velocity,-2.)",
            gradient="g_v=-2*v^-3*g_m", tangent="d_m=-2*v^-3*d_v",
            hash_encoding="flat C-order little-endian float64; velocity and native m have separate hashes",
            adapter_source_sha256=self._source_sha, native_oracle_source_sha256=NATIVE_SOURCE_SHA256)
        contract = native.contract
        identity = {key: contract[key] for key in ["operator_base_sha256", "geometry_sha256", "data_sha256",
                                                  "survey_sha256", "reference_source_sha256"]}
        self._problem = _json_sha(dict(native=identity, parameterization=parameterization))
        self._contract = dict(parameterization=parameterization, native=contract,
                              problem_fingerprint_sha256=self._problem,
                              optimizer_model_semantics="receipt.model, checkpoint.model and final_model are VELOCITY")
        _CLAIMED_NATIVES.add(native)

    @property
    def n(self):
        return self._native.n

    @property
    def contract(self):
        return json.loads(json.dumps(self._contract))

    @property
    def initial_model_record(self):
        return None if self._initial is None else dict(self._initial)

    def _count(self, name, amount=1):
        self._cost[name] = self._cost.get(name, 0) + int(amount)

    def cost_snapshot(self):
        values = self._native.cost_snapshot()
        values["velocity_coordinates"] = dict(self._cost)
        return values

    def _source_check(self):
        self._count("source_validation_attempts")
        adapter, native = self._source_path.read_bytes(), self._native_path.read_bytes()
        self._count("source_validation_bytes", len(adapter) + len(native))
        if _sha(adapter) != self._source_sha or _sha(native) != NATIVE_SOURCE_SHA256:
            raise RuntimeError("velocity adapter/native source changed after construction")

    def _convert(self, velocity):
        self._count("velocity_to_slowness_attempts")
        try:
            values = _immutable(_vector(velocity, size=self.n))
            model = velocity_to_slowness(values)
        except Exception:
            self._count("failed_coordinate_operations")
            raise
        self._count("converted_model_entries", self.n)
        self._count("canonical_copy_bytes", values.nbytes + model.nbytes)
        return values, model

    def to_native_model(self, velocity):
        """Paid coordinate conversion for future endpoint/API adapters; no solve."""
        self._source_check()
        values, model = self._convert(velocity)
        self._count("endpoint_conversions")
        return model, dict(velocity_model_sha256=model_sha256(values), actual_native_model_sha256=model_sha256(model),
                          problem_fingerprint_sha256=self._problem, parameterization="physical_velocity")

    def _key(self, velocity_sha, native):
        return _json_sha(dict(velocity_sha256=velocity_sha, native_receipt_key=native.key, problem=self._problem))

    def evaluate(self, velocity):
        self._source_check()
        values, model = self._convert(velocity)
        native = self._native.evaluate(model)  # Always dispatch, including genuine native cache hits.
        if model_sha256(model) != native.model_sha256 or not np.array_equal(model, native.model):
            raise ValueError("native receipt does not match the actual converted model")
        velocity_sha = model_sha256(values)
        key = self._key(velocity_sha, native)
        self._count("evaluations")
        for receipt in [self._current, self._trial]:
            if receipt is not None and receipt.key == key:
                self._check(receipt)
                self._count("velocity_receipt_cache_hits")
                return receipt
        receipt = VelocityMeritReceipt(native.objective, key, velocity_sha, values, self._problem, native, self._token)
        self._issued[receipt] = (values, native, _native_fingerprint(native), key, velocity_sha)
        self._trial = receipt
        if self._initial is None:
            self._initial = dict(velocity_model_sha256=velocity_sha, actual_native_model_sha256=native.model_sha256,
                                 receipt_key=key, problem_fingerprint_sha256=self._problem,
                                 original_input_slowness_bitwise_identity="not asserted; original slowness is not an optimizer input")
        return receipt

    def _check(self, receipt):
        self._source_check()
        self._count("receipt_validation_attempts")
        if type(receipt) is not VelocityMeritReceipt or receipt._owner is not self._token or receipt not in self._issued:
            raise ValueError("foreign or unissued velocity receipt")
        model, native, fingerprint, key, velocity_sha = self._issued[receipt]
        _, actual = self._convert(receipt.model)
        if (receipt.model is not model or receipt._native is not native or _native_fingerprint(native) != fingerprint
                or receipt.key != key or receipt.model_sha256 != velocity_sha or model_sha256(receipt.model) != velocity_sha
                or receipt.objective != native.objective or receipt.problem_fingerprint_sha256 != self._problem
                or receipt.key != self._key(velocity_sha, native) or model_sha256(actual) != native.model_sha256
                or model_sha256(native.model) != native.model_sha256 or not np.array_equal(actual, native.model)):
            raise ValueError("velocity/native receipt fingerprint mismatch")

    def accept(self, receipt):
        self._check(receipt)
        self._native.accept(receipt._native)
        self._current, self._trial = receipt, None
        self._count("accepted_receipts")

    def _factor(self, velocity):
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            factor = -2. * np.power(velocity, -3.)
        if not np.isfinite(factor).all() or not (factor < 0).all():
            self._count("failed_coordinate_operations")
            raise FloatingPointError("inverse-square derivative is not representable as finite negative float64")
        return factor

    def gradient(self, receipt):
        self._check(receipt)
        self._count("gradient_pullback_attempts")
        result = self._native.gradient(receipt._native)
        values = self._factor(receipt.model) * result.gradient
        if not np.isfinite(values).all():
            self._count("failed_coordinate_operations")
            raise FloatingPointError("velocity gradient is nonfinite")
        self._count("gradient_pullbacks")
        self._count("gradient_pullback_entries", self.n)
        return VelocityGradientEvaluation(result.objective, _immutable(values), receipt,
                                           result.additional_rhs, dict(result.frequency_strategies))

    def pushforward_direction(self, receipt, velocity_direction):
        """Map a v direction to native m; deliberately performs no tangent solve."""
        self._check(receipt)
        self._count("direction_pushforward_attempts")
        direction = _vector(velocity_direction, size=self.n, positive=False)
        values = self._factor(receipt.model) * direction
        if not np.isfinite(values).all():
            self._count("failed_coordinate_operations")
            raise FloatingPointError("native direction is nonfinite")
        self._count("direction_pushforwards")
        self._count("direction_pushforward_entries", self.n)
        return _immutable(values)
