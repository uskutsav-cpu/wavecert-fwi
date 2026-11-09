"""Exploratory old-endpoint diagnostic; no PDE or optimization execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def model_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def assess(repo, output, registration):
    if output.exists():
        raise FileExistsError("fresh diagnostic output required")
    rules = load(registration)
    inputs = {}
    def checked(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError("frozen old artifact changed: " + str(path))
        inputs[str(path.relative_to(repo))] = actual
        return path
    witness = repo / rules["witness_directory"]
    cohort = repo / rules["cohort_directory"]
    wreceipt = load(checked(witness / "complete_receipt.json", rules["witness_complete_sha256"]))
    receipt = load(checked(cohort / "complete_receipt.json", rules["cohort_complete_sha256"]))
    assert len(receipt["runs"]) == 48
    for delta, expected in [(0., 1.), (.5, .5), (4., 0.)]:
        matrix = np.diag([1. - delta, 3.])
        lower = max(0., 1. - delta)
        assert lower == expected and lower <= np.linalg.svd(matrix, compute_uv=False)[-1]
    rows, anchors = [], []
    for case in range(8):
        name = f"case_{case:02d}_receipt.json"
        record = load(checked(witness / name, wreceipt["artifact_sha256"][name]))
        raw_path = checked(witness / record["raw_arrays_artifact"], record["raw_arrays_file_sha256"])
        with np.load(raw_path, allow_pickle=False) as archive:
            initial = archive["initial_model"].copy()
        certificate = record["production"]["certificate"]
        assert model_sha(initial) == certificate["binding"]["model"]["sha256"]
        frequencies = certificate["frequency_records"]
        assert [row["frequency_hz"] for row in frequencies] == rules["frequencies_hz"]
        for frequency in frequencies:
            stability = frequency["stability"]
            assert stability["model_sha256"] == model_sha(initial)
            assert stability["physics_config"] == rules["physics"]
            beta = stability["beta_lower_bound"] if stability["available"] else None
            if beta is not None:
                assert math.isfinite(beta) and beta > 0
            omega2 = (2 * math.pi * frequency["frequency_hz"]) ** 2
            anchors.append(dict(case_index=case, frequency_hz=frequency["frequency_hz"],
                                beta_anchor=beta, positive_reuse_radius=None if beta is None else beta / omega2))
        endpoints = [row for row in receipt["runs"] if row["case_index"] == case]
        assert len(endpoints) == 6 and all(row["status"] == "complete" for row in endpoints)
        for endpoint in endpoints:
            model_path = checked(cohort / endpoint["models_artifact"], endpoint["models_sha256"])
            with np.load(model_path, allow_pickle=False) as archive:
                model = archive["final_native_m"].copy()
            assert model.shape == initial.shape and np.isfinite(model).all() and (model > 0).all()
            assert model_sha(model) == endpoint["output_provenance"]["final_actual_native_m_sha256"]
            distance = float(np.max(np.abs(model - initial)))
            bounds = []
            for frequency in frequencies:
                stability = frequency["stability"]
                beta = stability["beta_lower_bound"] if stability["available"] else None
                penalty = (2 * math.pi * frequency["frequency_hz"]) ** 2 * distance
                bounds.append(dict(frequency_hz=frequency["frequency_hz"], beta_anchor=beta,
                                   perturbation_penalty=penalty,
                                   candidate_lower_bound=None if beta is None else max(0., beta - penalty)))
            rows.append(dict(case_index=case, run_index=endpoint["run_index"], method=endpoint["method"],
                             repetition=endpoint["repetition"], model_displacement_infinity=distance,
                             frequency_bounds=bounds))
    assert len(rows) == 48 and len(anchors) == 32
    values = [item["candidate_lower_bound"] for row in rows for item in row["frequency_bounds"]]
    result = dict(role=rules["role"], rows=rows, anchors=anchors, endpoint_denominator=48,
                  frequency_endpoint_denominator=192, unavailable_anchor_instances=sum(v is None for v in values),
                  positive_instances=sum(v is not None and v > 0 for v in values),
                  zero_instances=sum(v == 0 for v in values), new_PDE_solves=0, new_optimizer_trajectories=0,
                  source_sha256=sha(Path(__file__)), registration_sha256=sha(registration),
                  input_sha256=inputs, limits=rules["limits"])
    assert all(sha(repo / path) == expected for path, expected in inputs.items())
    output.mkdir()
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["endpoint_denominator", "frequency_endpoint_denominator",
                                                "positive_instances", "zero_instances", "unavailable_anchor_instances"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    assess(args.repo.resolve(), args.output.resolve(), args.registration.resolve())
