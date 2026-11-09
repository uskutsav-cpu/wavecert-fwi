"""Copy exact prior48 retained initial-vector bytes into an explicit input; no PDE."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    old = root/"work/prior48_remote_study_01/extracted/study"
    inputs = json.loads((here/"RETAINED_INPUTS_01.json").read_bytes())
    prior = json.loads((old/"complete_receipt.json").read_bytes())
    initial, provenance = [], []
    for case in inputs["cases"]:
        row = next(r for r in prior["runs"] if r["case_index"] == case["case_index"])
        raw = (old/row["return_artifact"]).read_bytes()
        assert sha(raw) == row["return_sha256"]
        vector = np.asarray(json.loads(gzip.decompress(raw))["initial_model"], dtype="<f8")
        assert vector.shape == (2401,) and np.isfinite(vector).all() and (vector > 0).all()
        vector_sha = sha(vector.tobytes())
        assert vector_sha == case["old_linux_binding"]["initial_native_m_sha256"]
        initial.append(vector)
        provenance.append({"case_index": case["case_index"], "identity": case["identity"],
            "return_artifact": row["return_artifact"], "return_artifact_sha256": sha(raw),
            "initial_m_C_float64_sha256": vector_sha})
    output = here/"frozen_initials.npz"
    with output.open("xb") as stream:
        np.savez_compressed(stream, initial_m=np.stack(initial), case_indices=np.arange(8, dtype="<i8"),
            shape=np.array([49, 49], dtype="<i8"))
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {"initial_m", "case_indices", "shape"}
        assert all(sha(archive["initial_m"][i].tobytes()) == row["initial_m_C_float64_sha256"]
                   for i, row in enumerate(provenance))
        schema = {key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype),
            "payload_sha256": sha(archive[key].tobytes())} for key in archive.files}
    result = {"status": "EXACT_RETAINED_INITIALS_FROZEN_AS_EXPLICIT_INPUT_NO_PDE",
        "source_sha256": sha(Path(__file__).read_bytes()), "npz_sha256": sha(output.read_bytes()),
        "npz_bytes": output.stat().st_size, "schema": schema, "case_provenance": provenance,
        "truth49_arrays_present_in_exact_prior48_packet": False,
        "cost_scope": "Original optimistic truth-smoothed Gaussian initializer is a precomputed input, not a free fresh construction. This extraction cost/prehistory is separate; futureperrun loading/hashing/copying and allfresh observations are paid frombeforePopen. No oldtimings are pooled and no currentprotocol/API is changed; futurecaller needs reviewedinput/protocolamendment.",
        "data_ancestry": "Same reused eightSubsurfaceGen-derived development slices; source revision/license/transformations asbound inputreport. Initialvectors inheritthatrole and CC-BY-4.0 declared derivativeprovenance; no new rawsource/licenseclaim.",
        "PDE_calls": 0, "GO_created": False, "scientific_goals_complete": False}
    receipt = here/"FROZEN_INITIAL_INPUT_RECEIPT.json"
    with receipt.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt_sha256": sha(receipt.read_bytes()), "npz_sha256": result["npz_sha256"],
        "npz_bytes": result["npz_bytes"], "cases": 8, "PDE_calls": 0}))


if __name__ == "__main__":
    main()
