"""Independent analytic scalar checks on manufactured arrays only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE_SHA256 = "4aa248fd3862533390ad918f4ad3f70d02fcdebcdc8d784494d375c935536163"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(BASE / "fractional_velocity_prior.py") == SOURCE_SHA256
    spec = importlib.util.spec_from_file_location("independent_fractional_prior", BASE / "fractional_velocity_prior.py")
    component = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = component
    spec.loader.exec_module(component)
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    for shape in ((1, 1), (1, 4), (4, 1), (3, 4)):
        initial = np.array([[0.19 + 0.03*r + 0.02*c for c in range(shape[1])] for r in range(shape[0])])
        model = np.array([[float(initial[r, c]) / (1.03+0.04*r-0.02*c)**2 for c in range(shape[1])] for r in range(shape[0])])
        values = {(r, c): math.sqrt(float(initial[r, c]) / float(model[r, c])) - 1 for r in range(shape[0]) for c in range(shape[1])}
        weight = 9.0
        total = [q*q for q in values.values()]
        for (r, c), q in values.items():
            for other in ((r+1, c), (r, c+1)):
                if other in values:
                    total.append(weight * (q-values[other])**2)
        value = math.fsum(total) / (2*model.size)
        reference_gradient = np.zeros(shape)
        for (r, c), q in values.items():
            neighbors = [values[pos] for pos in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if pos in values]
            derivative_q = q + weight * math.fsum(q-other for other in neighbors)
            derivative_m = -math.sqrt(float(initial[r,c])) / (2*float(model[r,c])*math.sqrt(float(model[r,c])))
            reference_gradient[r,c] = derivative_q * derivative_m / model.size
        prior = component.FractionalVelocityPrior(initial, 0.1, 0.3)
        result = prior.evaluate(model)
        check(math.isclose(result.value, value, rel_tol=4e-13, abs_tol=2e-15), f"{shape} scalar edge energy")
        check(np.allclose(result.gradient, reference_gradient, rtol=4e-13, atol=3e-14), f"{shape} scalar analytic chain/graph gradient")
        zero = prior.evaluate(initial)
        check(zero.value == 0 and np.array_equal(zero.gradient, np.zeros(shape)), f"{shape} exact initial zero")
        equal_ratio = component.FractionalVelocityPrior(initial, 100, 300).evaluate(model)
        check(np.allclose(equal_ratio.gradient, result.gradient, rtol=2e-15, atol=2e-15) and math.isclose(equal_ratio.value, result.value, rel_tol=2e-15), f"{shape} matched length units")
    single = component.FractionalVelocityPrior([[0.25]], 1).evaluate([[0.16]])
    check(single.value == 0.03125 and single.roughness == 0 and single.gradient[0,0] == -0.9765625, "exact one-node closed form")

    prior = component.FractionalVelocityPrior(np.full((2, 2), 0.25), 1)
    prior.evaluate = lambda _model: (_ for _ in ()).throw(AssertionError("lambda0 evaluated prior"))
    initial_gradient = np.array([[1., -2.], [3., 4.]])
    result = component.normalized_objective(3, initial_gradient, 2, np.full((2,2),0.2), prior, 0)
    check(result.value == 1.5 and np.array_equal(result.gradient, initial_gradient / 2) and result.prior is None, "lambda0 never calls prior and retains shared normalization")
    check(np.array_equal(initial_gradient, [[1.,-2.],[3.,4.]]), "composer does not mutate input gradient")
    actions = [
        ("nonfinite input", lambda: component.FractionalVelocityPrior([[math.nan]], 1)),
        ("nonpositive model", lambda: component.FractionalVelocityPrior([[0.25]], 1).evaluate([[0]])),
        ("complex input", lambda: component.FractionalVelocityPrior([[1+1j]], 1)),
        ("boolean input", lambda: component.FractionalVelocityPrior([[True]], 1)),
        ("overflowed penalty", lambda: component.FractionalVelocityPrior([[1e300]], 1).evaluate([[1e-300]])),
        ("overflowed edge ratio", lambda: component.FractionalVelocityPrior([[1.0]], 1e-200, 1e200)),
        ("zero normalization", lambda: component.normalized_objective(1, [[0]], 0, [[0.25]], component.FractionalVelocityPrior([[0.25]],1), 0)),
        ("overflowed normalization", lambda: component.normalized_objective(1, [[1]], 1e-320, [[0.25]], component.FractionalVelocityPrior([[0.25]],1), 0)),
    ]
    for name, action in actions:
        try:
            action()
        except (ValueError, OverflowError, FloatingPointError):
            checks.append("reject " + name)
        else:
            raise AssertionError("accepted " + name)
    author_receipt = json.loads((BASE / "COMPONENT_VALIDATION_RECEIPT.json").read_bytes())
    for name, info in author_receipt["files"].items():
        check(sha(BASE / name) == info["sha256"] and (BASE/name).stat().st_size == info["bytes"], "author receipt exact asset " + name)
    for attempt, failures in (("01",1),("02",0)):
        tree = ET.parse(BASE / f"controls_attempt_{attempt}.xml")
        suite = tree.getroot().find("testsuite")
        check(int(suite.attrib["tests"]) == 39 and int(suite.attrib["failures"]) == failures and int(suite.attrib["errors"]) == 0 and int(suite.attrib["skipped"]) == 0, "actual author JUnit attempt " + attempt)
    receipt = {"status":"INDEPENDENT_MANUFACTURED_MATH_SOURCE_REVIEW_CLEAR", "reviewer":"geometry_acquisition_review agent", "source_sha256":SOURCE_SHA256, "review_control_source_sha256":sha(Path(__file__)), "checks_count":len(checks), "checks":checks, "largest_manufactured_array_entries":12, "project_arrays_decoded":0, "project_PDE_calls":0, "ordinary_float64_only":True, "author_sources_modified":False}
    (HERE / "INDEPENDENT_MANUFACTURED_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("status","checks_count","source_sha256","project_PDE_calls")}))


if __name__ == "__main__":
    main()
