"""Describe local/retained initial-vector differences; no criterion or PDE."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def f64_sha(value):
    return sha(np.ascontiguousarray(value, dtype="<f8").tobytes())


def main():
    here = Path(__file__).resolve().parent
    inspection = json.loads((here/"RETAINED_INPUTS_01.json").read_bytes())
    old = ROOT/"work/prior48_remote_study_01/extracted/study"
    receipt = json.loads((old/"complete_receipt.json").read_bytes())
    with np.load(ROOT/"work/wavecert-fwi/data/reference/subsurfacegen_parent_split/validation.npz", allow_pickle=False) as bundle:
        models = bundle["models"]
    rows = []
    for case in inspection["cases"]:
        old_run = next(row for row in receipt["runs"] if row["case_index"] == case["case_index"])
        raw = (old/old_run["return_artifact"]).read_bytes()
        assert sha(raw) == old_run["return_sha256"]
        payload = json.loads(gzip.decompress(raw))
        retained = np.asarray(payload["initial_model"], dtype=np.float64)
        assert f64_sha(retained) == case["old_linux_binding"]["initial_native_m_sha256"]
        v = 1/np.sqrt(models[case["identity"]["array_index"]].astype(np.float64))
        nodes = np.arange(1, 50)/50*25
        zz, xx = np.meshgrid(nodes, nodes, indexing="ij")
        fine = RegularGridInterpolator((np.arange(26, dtype=float),)*2, np.pad(v, 1, mode="edge"))(
            np.stack((zz, xx), axis=-1))
        local = (1/gaussian_filter(fine, sigma=.08/.025, mode="reflect")**2).reshape(-1)
        assert f64_sha(local) == case["local_derived_initial49_sha256"]
        difference = local-retained
        rows.append({"case_index": case["case_index"], "return_artifact": old_run["return_artifact"],
            "return_artifact_sha256": sha(raw), "retained_initial_sha256": f64_sha(retained),
            "local_initial_sha256": f64_sha(local), "different_elements": int(np.count_nonzero(difference)),
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "relative_l2_difference": float(np.linalg.norm(difference)/np.linalg.norm(retained)),
            "max_difference_in_retained_float_spacings": float(np.max(np.abs(difference)/np.spacing(retained)))})
    result = {"status": "INITIAL_VECTOR_DIFFERENCES_DESCRIBED_NOT_A_NEW_ACCEPTANCE_TEST",
        "inspection_source_sha256": sha(Path(__file__).read_bytes()),
        "retained_input_report_sha256": sha((here/"RETAINED_INPUTS_01.json").read_bytes()),
        "cases": rows, "max_absolute_difference": max(row["max_absolute_difference"] for row in rows),
        "max_relative_l2_difference": max(row["relative_l2_difference"] for row in rows),
        "limits": "Exact archived initial hashes verified; local reconstruction not byteidentical. Numerical differences are descriptive, not a posthoc equivalence or physics criterion. Cause not isolated: local and archived numerical runtimes/platforms differ. No source/API/input/initializer change, no oldinitial forced into futureworker, no PDE or observation solve.",
        "PDE_calls": 0, "scientific_goals_complete": False}
    output = here/"INITIAL_DIFFERENCES_01.json"
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": sha(output.read_bytes()), "cases": len(rows),
        "max_absolute_difference": result["max_absolute_difference"],
        "max_relative_l2_difference": result["max_relative_l2_difference"], "PDE_calls": 0}))


if __name__ == "__main__":
    main()
