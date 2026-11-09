"""Data-only scalar normalization of real issued receipts; no prior or solver."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

_LIVE_BINDINGS = []


def immutable_model(value):
    value = np.asarray(value)
    if (value.dtype != np.float64 or value.ndim != 1 or not value.size
            or not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError("finite positive flat float64 direct m required")
    return np.frombuffer(value.tobytes(), dtype=np.float64)


def model_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def count_tree(value):
    if not isinstance(value, dict):
        raise TypeError("nested integer cost dictionary required")
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = count_tree(item)
        elif type(item) is int and item >= 0:
            result[key] = item
        else:
            raise ValueError("cost entries must be nonnegative actual integer counts")
    return result


def _zero(value):
    return all(_zero(v) if isinstance(v, dict) else v == 0 for v in value.values())


@dataclass(frozen=True, eq=False)
class NormalizedReceipt:
    objective: float
    model: np.ndarray = field(repr=False)
    raw: object = field(repr=False)
    owner: object = field(repr=False)
    model_sha256: str
    raw_objective: float


class NormalizedExactOracle:
    """Keep authentic native objects, unwrap only locally issued receipts.

    The caller must independently bind the real native class/module source.
    Manufactured controls use a distinct declared toy issuer. This wrapper
    neither fabricates native headers nor substitutes numerical functions.
    """
    def __init__(self, native, initial_model):
        _LIVE_BINDINGS[:] = [(owner, policy) for owner, policy in _LIVE_BINDINGS
                             if owner() is not None and policy() is not None]
        if any(owner() is native for owner, _ in _LIVE_BINDINGS):
            raise ValueError("native owner already belongs to a live normalized policy")
        self.native = native
        self.initial_model = immutable_model(initial_model)
        self.token, self.issued = object(), weakref.WeakSet()
        self._phi0 = self.initial_header = self.current = None
        self.counts = {"initial_model_copied_bytes": self.initial_model.nbytes}
        self.native_prefix = None
        self.unknown_work = self.failed_dispatch = False
        self.last_failure = None
        self._capture()
        if self.unknown_work or not _zero(self.native_prefix):
            raise ValueError("a known cold native owner is required")
        _LIVE_BINDINGS.append((weakref.ref(native), weakref.ref(self)))

    @property
    def phi0(self):
        return self._phi0

    def _count(self, key, amount=1):
        self.counts[key] = self.counts.get(key, 0) + amount

    def _capture(self):
        self._count("native_snapshot_attempts")
        try:
            self.native_prefix = count_tree(copy.deepcopy(self.native.cost_snapshot()))
            self._count("native_snapshots_completed")
            self.unknown_work = self.failed_dispatch
        except BaseException as exc:  # noqa: BLE001 -- preserve unknown failed tails
            self.unknown_work = True
            self.last_failure = {"phase": "snapshot", "exception_type": type(exc).__name__}

    def _call(self, name, *args):
        self._count(name + "_attempts")
        self.unknown_work = True
        completed = False
        try:
            result = getattr(self.native, name)(*args)
            self._count(name + "_completed")
            completed = True
            return result
        except BaseException as exc:
            self.last_failure = {"phase": name, "exception_type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            if not completed:
                self.failed_dispatch = True
            self._capture()

    def evaluate(self, model):
        model = immutable_model(model)
        self._count("requested_model_copied_bytes", model.nbytes)
        if self.phi0 is None and not np.array_equal(model, self.initial_model):
            raise ValueError("first paid model must be the direct initializer")
        raw = self._call("evaluate", model)
        if (np.asarray(raw.model).dtype != np.float64 or not np.array_equal(raw.model, model)
                or not math.isfinite(float(raw.objective)) or raw.objective < 0):
            raise ValueError("native receipt does not match requested finite direct model/Phi")
        phi = float(raw.objective)
        if self.phi0 is None:
            if phi <= 0:
                raise ValueError("positive paid initial Phi0 is required")
            self._phi0 = phi
            self.initial_header = {"model_sha256": model_sha(model), "raw_objective": phi,
                "raw_binding": {name: getattr(raw, name) for name in (
                    "key", "model_sha256", "operator_sha256", "geometry_sha256", "data_sha256")}}
        receipt = NormalizedReceipt(phi / self.phi0, model, raw, self.token, model_sha(model), phi)
        self.issued.add(receipt)
        self._count("scalar_divisions")
        return receipt

    def _validate(self, receipt):
        if (type(receipt) is not NormalizedReceipt or receipt.owner is not self.token
                or receipt not in self.issued or model_sha(receipt.model) != receipt.model_sha256
                or not np.array_equal(receipt.model, receipt.raw.model)
                or receipt.raw_objective != receipt.raw.objective
                or receipt.objective != receipt.raw_objective / self.phi0):
            raise ValueError("foreign, forged or altered normalized receipt")

    def accept(self, receipt):
        self._validate(receipt)
        self._call("accept", receipt.raw)
        self.current = receipt

    def gradient(self, receipt):
        self._validate(receipt)
        result = self._call("gradient", receipt.raw)
        gradient = np.asarray(result.gradient)
        if (gradient.dtype != np.float64 or gradient.shape != receipt.model.shape
                or not np.isfinite(gradient).all()):
            raise ValueError("invalid genuine native gradient")
        scaled = gradient / self.phi0
        self._count("gradient_real_divisions", gradient.size)
        self._count("gradient_allocated_bytes", scaled.nbytes)
        return SimpleNamespace(gradient=scaled)

    def cost_snapshot(self):
        """No query: cached cumulative prefix once, alongside wrapper counters."""
        return {"native": copy.deepcopy(self.native_prefix), "normalization": dict(self.counts)}

    def diagnostics(self):
        result = {"Phi0": self.phi0, "initial_header": self.initial_header,
            "current_model_sha256": None if self.current is None else self.current.model_sha256,
            "native_last_known_cumulative": copy.deepcopy(self.native_prefix),
            "unknown_work": self.unknown_work, "failed_dispatch_work_may_exceed_prefix": self.failed_dispatch,
            "counts": dict(self.counts), "last_failure": copy.deepcopy(self.last_failure)}
        json.dumps(result, allow_nan=False)
        return result
