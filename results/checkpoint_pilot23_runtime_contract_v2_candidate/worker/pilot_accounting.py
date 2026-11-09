"""Small dependency-free checks for a held ordinary acoustic replay control."""
from __future__ import annotations

import hashlib
import json
import math
import time


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


class Ledger:
    def __init__(self, path=None, maximum_events=8192):
        self.path = path
        self.maximum_events = maximum_events
        self.events = []
        self.phase = "setup"
        self.started = time.perf_counter()

    def add(self, kind, **values):
        if len(self.events) >= self.maximum_events:
            raise RuntimeError("fixed event cap exceeded; no unrecorded native call admitted")
        event = {"sequence": len(self.events), "phase": self.phase,
                 "seconds_since_ledger": time.perf_counter()-self.started,
                 "kind": kind, **values}
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
                stream.flush()
        return event


def checkpoint_budget(size_elements, itemsize, checkpoints, cap_bytes):
    values = (size_elements, itemsize, checkpoints, cap_bytes)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("explicit positive integer element/item/count/cap values required")
    one = size_elements * itemsize
    total = checkpoints * one
    if total > cap_bytes:
        raise ValueError("explicit checkpoint capacity exceeds the registered byte cap")
    return {"size_elements": size_elements, "itemsize": itemsize,
            "checkpoint_bytes": one, "checkpoints": checkpoints,
            "capacity_bytes": total, "cap_bytes": cap_bytes}


class AuditedKernel:
    """Observe the real wrapper; constructor never advances the operator."""
    def __init__(self, native, role, steps, ledger, immutable):
        self.native, self.role, self.steps = native, role, steps
        self.ledger, self.immutable = ledger, immutable
        self.original = {name: supplier() for name, supplier in immutable.items()}
        self.start_offset = int(native.start_offset)
        self.calls = []

    def apply(self, t_start, t_end):
        if (type(t_start) is not int or type(t_end) is not int
                or not 0 <= t_start < t_end <= self.steps):
            raise ValueError("invalid scheduled half-open kernel range")
        for name, supplier in self.immutable.items():
            if supplier() != self.original[name]:
                raise ValueError("immutable input changed before native call: "+name)
        translated = self.native._prepare_args(t_start, t_end)
        time_m, time_M = int(translated["time_m"]), int(translated["time_M"])
        if (time_m, time_M) != (t_start+self.start_offset, t_end-1+self.start_offset):
            raise ValueError("native time translation differs from the pinned wrapper")
        row = {"role": self.role, "scheduled_start": t_start, "scheduled_end": t_end,
               "native_time_m": time_m, "native_time_M": time_M,
               "native_steps": t_end-t_start, "status": "attempted"}
        self.calls.append(row)
        self.ledger.add("native_attempt", **row)
        before = time.perf_counter()
        try:
            self.native.apply(t_start=t_start, t_end=t_end)
        except BaseException as error:
            row.update(status="raised", seconds=time.perf_counter()-before,
                       error=type(error).__name__+": "+str(error))
            self.ledger.add("native_raised", **row)
            raise
        row.update(status="returned", seconds=time.perf_counter()-before)
        self.ledger.add("native_returned", **row)
        for name, supplier in self.immutable.items():
            if supplier() != self.original[name]:
                self.ledger.add("immutable_input_changed", name=name, role=self.role)
                raise ValueError("immutable input changed during native call: "+name)

    def counts(self):
        return {"attempted_calls": len(self.calls),
                "returned_calls": sum(x["status"] == "returned" for x in self.calls),
                "attempted_steps": sum(x["native_steps"] for x in self.calls),
                "returned_steps": sum(x["native_steps"] for x in self.calls if x["status"] == "returned")}


class AuditedStorage:
    """Count actual supplied pointers and delegate unchanged precision copies."""
    def __init__(self, native, checkpoint_bytes, ledger):
        self.native, self.checkpoint_bytes, self.ledger = native, checkpoint_bytes, ledger
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _copy(self, operation, key, pointers):
        pointers = list(pointers)
        actual_bytes = sum(int(p.nbytes) for p in pointers)
        if actual_bytes != self.checkpoint_bytes:
            raise ValueError("actual checkpoint pointer bytes differ from size*dtype.itemsize")
        row = {"operation": operation, "key": int(key), "pointer_count": len(pointers),
               "bytes": actual_bytes, "status": "attempted"}
        self.calls.append(row)
        self.ledger.add("checkpoint_copy_attempt", **row)
        before = time.perf_counter()
        try:
            getattr(self.native, operation)(key, pointers)
        except BaseException as error:
            row.update(status="raised", seconds=time.perf_counter()-before,
                       error=type(error).__name__+": "+str(error))
            self.ledger.add("checkpoint_copy_raised", **row)
            raise
        row.update(status="returned", seconds=time.perf_counter()-before)
        self.ledger.add("checkpoint_copy_returned", **row)

    def save(self, key, pointers):
        return self._copy("save", key, pointers)

    def load(self, key, locations):
        return self._copy("load", key, locations)


def edge_pullback(padded, shape, nbl):
    """E^T for replicated physical edges; padded is a 2D iterable, no halo."""
    nx, nz = shape
    if len(padded) != nx+2*nbl or any(len(row) != nz+2*nbl for row in padded):
        raise ValueError("padded grid differs from the registered extension")
    result = [[0.0 for _ in range(nz)] for _ in range(nx)]
    for i, row in enumerate(padded):
        ii = min(max(i-nbl, 0), nx-1)
        for j, value in enumerate(row):
            jj = min(max(j-nbl, 0), nz-1)
            result[ii][jj] += float(value)
    return result


def centered_difference(plus, minus, epsilon):
    if (not all(math.isfinite(v) for v in (plus, minus, epsilon)) or epsilon <= 0):
        raise ValueError("finite objectives and positive epsilon required")
    return (plus-minus)/(2*epsilon)


def relative_scalar_error(a, b):
    if not all(math.isfinite(v) for v in (a, b)):
        raise ValueError("nonfinite derivative")
    return abs(a-b)/max(abs(a), abs(b), 1e-30)
