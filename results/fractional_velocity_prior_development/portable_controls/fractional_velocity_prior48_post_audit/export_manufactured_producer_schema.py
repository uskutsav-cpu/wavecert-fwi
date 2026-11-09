"""TEST ONLY: real frozen engine/adapter export over a manufactured scalar.

No WaveCert module, project input, native PDE oracle or study runner is used.
Run this helper in a separate process so the auditor runtime stays independent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PINS = {
    "run_prior48.py": "7c7fc525894dbd38899ad6ee46bff8a0a0d996ea2ef61d5da2e6ce8b4320d882",
    "protocol.json": "92433c9d7b9d2990799f3a81707f6d04f63784a1820d3f109804a83b484ccfa6",
    "bounded_exact.py": "d5fb6326bcd252efc3a7672dba993311fb0ba724b90039e7fda317bcf8bb770c",
    "composed_exact_adapter.py": "28f8bc9e6e71fd8c28165a8eddcb56e605cd7323286578180fabdf351e75a637",
    "fractional_velocity_prior.py": "4aa248fd3862533390ad918f4ad3f70d02fcdebcdc8d784494d375c935536163",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def vector_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self):
        self.value += 0.001


@dataclass(eq=False)
class ScalarPoint:
    model: np.ndarray
    objective: float
    owner: object
    gradient_cached: bool = False

    def to_dict(self):
        model = vector_sha(self.model)
        operator = canonical_sha(dict(manufactured_scalar_operator_for=model))
        geometry = canonical_sha(dict(manufactured_dimension=1, no_wave_geometry=True))
        data = canonical_sha(dict(manufactured_scalar_target=0.1, no_observations=True))
        return dict(
            kind="manufactured_scalar_schema_receipt_NOT_NATIVE_PDE",
            objective=self.objective,
            model_sha256=model,
            operator_sha256=operator,
            geometry_sha256=geometry,
            data_sha256=data,
            key=canonical_sha(dict(operator=operator, geometry=geometry, data=data)),
            gradient_cached=self.gradient_cached,
            frequency_merit_routes={"manufactured_scalar": "no_PDE"},
            frequency_field_shapes={"manufactured_scalar": [1]},
        )


class ScalarOwner:
    def __init__(self, clock):
        self.clock, self.n = clock, 1
        self.current = self.trial = None
        self.issued = set()
        self.counts = dict(
            evaluate_dispatches=0,
            gradient_dispatches=0,
            actual_scalar_merit_computations=0,
            accepts=0,
        )
        clock.advance()

    def evaluate(self, model):
        self.counts["evaluate_dispatches"] += 1
        self.clock.advance()
        for point in (self.current, self.trial):
            if point is not None and np.array_equal(point.model, model):
                return point
        model = np.asarray(model, dtype=np.float64).copy()
        model.flags.writeable = False
        point = ScalarPoint(model, 0.5 * float((model[0] - 0.1) ** 2), self)
        self.counts["actual_scalar_merit_computations"] += 1
        self.issued.add(point)
        self.trial = point
        return point

    def gradient(self, point):
        if point.owner is not self or point not in self.issued:
            raise ValueError("manufactured scalar ownership mismatch")
        self.counts["gradient_dispatches"] += 1
        self.clock.advance()
        point.gradient_cached = True
        return SimpleNamespace(gradient=np.asarray(point.model - 0.1, dtype=np.float64))

    def accept(self, point):
        if point.owner is not self or point not in self.issued:
            raise ValueError("manufactured scalar ownership mismatch")
        self.counts["accepts"] += 1
        self.clock.advance()
        self.current, self.trial = point, None

    def cost_snapshot(self):
        return dict(manufactured_scalar=dict(self.counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=float, choices=[0.0, 100.0], required=True)
    args = parser.parse_args()
    folder = args.producer_directory.resolve()
    for name, expected in PINS.items():
        if sha(folder / name) != expected:
            raise ValueError("actual frozen producer pin changed: " + name)
    driver = load("_test_actual_frozen_prior48_driver", folder / "run_prior48.py")
    adapter = load("_test_actual_frozen_composed_adapter", folder / "composed_exact_adapter.py")
    p, _ = driver.load_protocol(folder / "protocol.json")  # Metadata/dependency hashes only.
    p = copy.deepcopy(p)
    p["physics"]["shape"] = [1, 1]  # TEST ONLY manufactured dimensional scope.
    dependencies = adapter.load_frozen_dependencies()
    from scipy.optimize import minimize

    if dependencies.engine.minimize is not minimize:
        raise ValueError("interoperability control requires the real SciPy minimize")
    runtime = SimpleNamespace(adapter=adapter, dependencies=dependencies, np=np)
    clock, owners, attempt = Clock(), [], {}

    def prepare():
        clock.advance()
        return SimpleNamespace(initial=np.array([0.25], dtype=np.float64))

    def make_native(_problem):
        owner = ScalarOwner(clock)
        owners.append(owner)
        return owner

    measurement = driver.execute_policy(
        prepare=prepare,
        make_native=make_native,
        runtime=runtime,
        p=p,
        weight=args.weight,
        check_static=clock.advance,
        attempt=attempt,
        clock=clock,
    )
    returned = clock()
    timing = driver.return_gate(measurement, returned, 8.0)
    payload = measurement["payload"]
    engine = payload["engine"]
    if (
        len(owners) != 1
        or not payload["method_valid"]
        or not timing["eligible"]
        or owners[0].counts["gradient_dispatches"] <= 0
        or engine["all_library_evaluation_trace"]["total_count"] <= 0
        or engine["native_callback_point_records"] <= 0
    ):
        raise ValueError("nonzero-query actual engine scalar export required")
    if any(name.startswith("wavecert") for name in sys.modules):
        raise ValueError("project physics module imported by synthetic control")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "actual_exported_payload.json").write_bytes(measurement["serialized"])
    metadata = dict(
        scope="TEST_ONLY_NONZERO_QUERY_SCHEMA_INTEROPERABILITY_NOT_NATIVE_PDE",
        producer_source_pins=PINS,
        weight=args.weight,
        native_owner="manufactured ScalarOwner, not ExactMeritOracle",
        dimension=1,
        actual_frozen_engine_called=True,
        actual_frozen_adapter_called=True,
        actual_driver_execute_policy_called=True,
        actual_SciPy_minimize_used=True,
        actual_scalar_queries=dict(owners[0].counts),
        library_events=engine["all_library_evaluation_trace"]["total_count"],
        callback_point_records=engine["native_callback_point_records"],
        payload_sha256=sha(args.output / "actual_exported_payload.json"),
        timing=timing,
        fake_clock_for_schema_only=True,
        project_array_reads=0,
        project_PDE_calls=0,
        fixture_files_sha256={
            name: sha(folder / name)
            for name in ["run_prior48.py", "protocol.json", *p["local_dependency_sha256"]]
        },
    )
    (args.output / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                scope=metadata["scope"],
                weight=args.weight,
                library_events=metadata["library_events"],
                callbacks=metadata["callback_point_records"],
            )
        )
    )


if __name__ == "__main__":
    main()
