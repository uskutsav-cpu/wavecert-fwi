"""Cold public receiver-map batches; frozen native binding is SOURCE ONLY.

No production imports, owners or PDEs are constructed at module import. The
manufactured controls supply a dense toy factory instead of native physics.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import math
import time
import weakref
from pathlib import Path

HELMHOLTZ_SHA256 = "caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131"
EXACT_MERIT_SHA256 = "0fec1f215b28e36017c1e28409d7f7e909a03cbf5b717de5d23eb8d52eb26810"
CURRENT_FREQUENCIES_HZ = (.25, .5, 1., 2., 2.5, 3., 3.5, 4.)


def current_full_layout():
    """Registered metadata only; constructs no model/data arrays or owner."""
    from cached_jacobian import Layout

    sources = (250, 258, 266, 272, 280, 288)
    receivers = tuple(range(250, 289, 2))
    return Layout(2401, receivers, tuple((frequency, sources, tuple(range(6*i, 6*i+6)))
        for i, frequency in enumerate(CURRENT_FREQUENCIES_HZ)))


class PublicReceiverMapBuilder:
    """One fresh public physics owner per missing-row batch, including failures."""
    def __init__(self, factory, *, validate=None, clock=time.perf_counter):
        self.factory, self.validate, self.clock = factory, validate, clock
        self.attempts = []
        # Native Helmholtz2D is an unhashable dataclass. Ownership is object
        # identity, never value equality/hash; weak refs must not retain LU.
        self.owners = []

    def build(self, *, point, frequency, receivers):
        started = self.clock()
        row = {"index": len(self.attempts), "point_identity": list(point.identity),
            "frequency_hz": frequency, "receivers": list(receivers), "factory_attempted": True,
            "factory_completed": False, "map_dispatch_attempted": False, "map_completed": False,
            "last_known_physics_prefix": None, "unknown_work": True}
        self.attempts.append(row)
        physics = None
        try:
            physics = self.factory(point.config)
            row["factory_completed"] = True
            self.owners = [reference for reference in self.owners if reference() is not None]
            if any(reference() is physics for reference in self.owners):
                raise ValueError("map factory reused a prior numerical owner")
            self.owners.append(weakref.ref(physics))
            if self.validate is not None:
                self.validate(physics, point)
            before = copy.deepcopy(physics.cost_snapshot())
            if any(isinstance(x, bool) or not isinstance(x, int) or x != 0 for x in before.values()):
                raise ValueError("map builder requires a new zero-work physics owner")
            row["before_physics"] = before
            row["cold_owner_verified"] = True
            row["map_dispatch_attempted"] = True
            # Public API only. No supplied inverse, no source/state re-solve,
            # no access to native receipt records or their private LU factors.
            result = physics.receiver_resolvent_matrix(point.model, frequency, receivers)
            row["map_completed"] = True
            after = copy.deepcopy(physics.cost_snapshot())
            row["last_known_physics_prefix"] = after
            expected = {key: 0 for key in before}
            expected.update(factorizations=1, stability_rhs=len(receivers))
            if after != expected:
                raise ValueError("actual public map work differs from one fresh LU/missing receiver batch")
            row["actual_complete_batch_counts_verified"] = True
            row["unknown_work"] = False
            return result
        except BaseException as exc:
            row["failure_type"] = type(exc).__name__
            # Public counters may increment only AFTER numerical calls; a
            # failed dispatch can have work beyond the observed prefix.
            row["unknown_work"] = True
            raise
        finally:
            if physics is not None:
                try:
                    row["last_known_physics_prefix"] = copy.deepcopy(physics.cost_snapshot())
                    row["final_snapshot_completed"] = True
                except BaseException as exc:  # noqa: BLE001 -- retain interrupted numerical prefix
                    row["final_snapshot_failure_type"] = type(exc).__name__
                    row["unknown_work"] = True
            row["elapsed_seconds"] = self.clock()-started

    def cost_snapshot(self):
        return {"cold_owner_batches": copy.deepcopy(self.attempts),
            "unknown_work_present": any(row["unknown_work"] for row in self.attempts),
            "scope": "One actual final physics prefix per distinct cold owner; no shared native factors or overlapping snapshot sums."}


def frozen_native_binding(native, *, reference_package_root, initial_model, layout):
    """SOURCE-ONLY integration entry; not called by controls/default entrypoint.

    A future separately approved integration may call it with a genuine cold
    ExactMeritOracle. It performs no data loading or observation synthesis.
    """
    from cached_jacobian import CachedJacobianCallbacks

    if layout != current_full_layout():
        raise ValueError("source-only native entry requires the full registered8-frequency/48-block/1920-real-term survey")
    root = Path(reference_package_root).resolve()
    helm = importlib.import_module("wavecert.physics.helmholtz")
    exact = importlib.import_module("wavecert.physics.exact_merit")
    for module, relative, expected in ((helm, "physics/helmholtz.py", HELMHOLTZ_SHA256),
            (exact, "physics/exact_merit.py", EXACT_MERIT_SHA256)):
        path = root/relative
        if Path(module.__file__).resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen public native source/origin mismatch")
    if type(native) is not exact.ExactMeritOracle:
        raise ValueError("actual frozen ExactMeritOracle required")
    for name in ("evaluate", "export_merit_fields", "validate_merit_fields", "cost_snapshot"):
        if getattr(getattr(native, name), "__func__", None) is not getattr(exact.ExactMeritOracle, name):
            raise ValueError("native public method changed")
    contract = native.contract
    config = contract["operator_base_record"]["configuration"]
    if config != {"shape": [49, 49], "spacing": .025, "damping_width": 6, "damping_strength": 2.}:
        raise ValueError("source-only native entry requires the registered49x49 physics")

    def factory(copied_config):
        shape, spacing, width, strength = copied_config
        if (list(shape), spacing, width, strength) != (config["shape"], config["spacing"], config["damping_width"], config["damping_strength"]):
            raise ValueError("native/map physics configuration mismatch")
        return helm.Helmholtz2D(shape=shape, spacing=spacing, damping_width=width,
            damping_strength=strength, factorization_cache_size=0)

    def validate(physics, point):
        if type(physics) is not helm.Helmholtz2D:
            raise ValueError("actual frozen cold map physics required")
        for name in ("operator", "receiver_resolvent_matrix", "cost_snapshot"):
            if (getattr(getattr(physics, name), "__func__", None) is not getattr(helm.Helmholtz2D, name)
                    or Path(inspect.getsourcefile(getattr(physics, name))).resolve() != root/"physics/helmholtz.py"):
                raise ValueError("map public method source changed")
        if (point.identity[3], point.identity[4]) != (contract["geometry_sha256"], contract["data_sha256"]):
            raise ValueError("copied point survey/data binding changed")
        expected_amplitude = 1/point.config[1]**2
        if not math.isfinite(expected_amplitude) or any(a != expected_amplitude for frame in point.frames for a in frame.amplitudes):
            raise ValueError("actual native point-source amplitude differs")

    return CachedJacobianCallbacks(native, PublicReceiverMapBuilder(factory, validate=validate),
        initial_model=initial_model, layout=layout)
