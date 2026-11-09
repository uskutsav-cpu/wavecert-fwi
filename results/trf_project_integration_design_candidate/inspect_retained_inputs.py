"""Read retained validation schema/ancestry and derive initializers; no PDE/importproducer."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT/"work/wavecert-fwi"
BASE = REPO/"data/reference/subsurfacegen_parent_split"
OLD = ROOT/"work/prior48_remote_study_01/extracted/study"
NEW = ROOT/"work/new-phase-staging/trf_matched_cost_driver_candidate"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def f64_sha(value):
    return sha(np.ascontiguousarray(value, dtype="<f8").tobytes())


def inspect():
    protocol_raw = (OLD/"protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    go_raw = (OLD/"root_go.json").read_bytes()
    go = json.loads(go_raw)
    old_driver = (OLD/"driver_source.py").read_bytes()
    assert sha(protocol_raw) == go["protocol_sha256"] == "92433c9d7b9d2990799f3a81707f6d04f63784a1820d3f109804a83b484ccfa6"
    assert sha(old_driver) == go["driver_sha256"] == "7c7fc525894dbd38899ad6ee46bff8a0a0d996ea2ef61d5da2e6ce8b4320d882"
    assert sha(go_raw) == "44234fe02f29956a74b61850e8cb7752841aff97545ea9c1e5e2039df5efa1e1"
    inputs = {}
    for relative, expected in protocol["inputs_sha256"].items():
        path = REPO/relative
        raw = path.read_bytes()
        assert sha(raw) == expected
        inputs[relative] = {"sha256": sha(raw), "bytes": len(raw)}
    metadata_raw = (BASE/"validation_metadata.csv").read_bytes()
    split_raw = (BASE/"split_manifest.csv").read_bytes()
    metadata = list(csv.DictReader(io.StringIO(metadata_raw.decode())))
    split = list(csv.DictReader(io.StringIO(split_raw.decode())))
    ancestry_path = REPO/"results/new_ancestry_audit/parent_ancestry.csv"
    ancestry = list(csv.DictReader(io.StringIO(ancestry_path.read_text())))
    prior = json.loads((OLD/"complete_receipt.json").read_bytes())
    assert prior["source_before"]["driver_sha256"] == go["driver_sha256"]
    new_protocol = json.loads((NEW/"protocol.json").read_bytes())
    schedule_raw = (NEW/"FROZEN48_SCHEDULE.json").read_bytes()
    schedule = json.loads(schedule_raw)
    assert sha(schedule_raw) == new_protocol["schedule_sha256"]
    assert protocol["identities"] == new_protocol["identities"]
    assert len(schedule) == 48 and all(row["status"] == "unattempted" for row in schedule)
    raw_npz = (BASE/"validation.npz").read_bytes()
    arrays, members = {}, {}
    with zipfile.ZipFile(io.BytesIO(raw_npz)) as container:
        assert len(container.infolist()) == 7
        for item in container.infolist():
            raw = container.read(item)
            members[item.filename] = {"NPY_sha256": sha(raw), "NPY_bytes": len(raw),
                "compressed_bytes": item.compress_size, "CRC32": f"{item.CRC:08x}"}
    with np.load(io.BytesIO(raw_npz), allow_pickle=False) as container:
        values = {key: container[key] for key in container.files}
    expected = {"models": ((192, 24, 24), "float32"), "source_indices": ((192,), "int64"),
        "frequencies_hz": ((192,), "float32"), "wavefields": ((192, 24, 24, 2), "float32"),
        "receiver_indices": ((20,), "int64"), "shape": ((2,), "int64"), "spacing": ((), "float64")}
    assert set(values) == set(expected) and len(metadata) == 192
    for key, array in values.items():
        assert (array.shape, str(array.dtype)) == expected[key] and np.isfinite(array).all()
        arrays[key] = {"shape": list(array.shape), "dtype": str(array.dtype), "nbytes": array.nbytes,
            "C_payload_sha256": sha(np.ascontiguousarray(array).tobytes()), "finite": True}
    assert (values["models"] > 0).all() and values["shape"].tolist() == [24, 24]
    assert float(values["spacing"]) == .05
    assert values["receiver_indices"].tolist() == list(range(50, 70))
    geometry = protocol["original_geometry_context"]
    pairs = {(s, f) for s in geometry["original_source_indices"] for f in geometry["frequencies_hz"]}
    cases = []
    for case, identity in enumerate(protocol["identities"]):
        rows = [i for i, row in enumerate(metadata) if row["slice_id"] == identity["slice_id"]]
        assert len(rows) == 24 and rows[0] == identity["array_index"]
        assert {(int(values["source_indices"][i]), float(values["frequencies_hz"][i])) for i in rows} == pairs
        assert all(metadata[i]["model_id"] == identity["parent_id"] and metadata[i]["split"] == "validation"
            and metadata[i]["source_file"] == identity["source_file"]
            and metadata[i]["source_sha256"] == identity["source_sha256"] for i in rows)
        assert all(np.array_equal(values["models"][i], values["models"][rows[0]]) for i in rows)
        manifest = [r for r in split if r["slice_id"] == identity["slice_id"]]
        assert len(manifest) == 1 and manifest[0]["split"] == "validation" and manifest[0]["model_id"] == identity["parent_id"]
        raw_m = values["models"][rows[0]].astype(np.float64)
        velocity = 1/np.sqrt(raw_m)
        nodes = np.arange(1, 50)/50*25
        zz, xx = np.meshgrid(nodes, nodes, indexing="ij")
        fine = RegularGridInterpolator((np.arange(26, dtype=float),)*2, np.pad(velocity, 1, mode="edge"))(
            np.stack((zz, xx), axis=-1))
        truth = (1/fine**2).reshape(-1)
        initial = (1/gaussian_filter(fine, sigma=.08/.025, mode="reflect")**2).reshape(-1)
        old_bindings = [r["common_binding"] for r in prior["runs"] if r["case_index"] == case]
        assert len(old_bindings) == 6 and all(x == old_bindings[0] for x in old_bindings)
        old_binding = old_bindings[0]
        assert f64_sha(raw_m) == old_binding["raw_model_sha256"]
        lo, hi = protocol["squared_slowness_bounds"]
        assert np.isfinite(initial).all() and np.isfinite(truth).all() and (truth > 0).all()
        margin = min(float(np.min(initial-lo)), float(np.min(hi-initial)))
        assert margin > 1e-10
        ancestors = [r for r in ancestry if r["model_id"] == identity["parent_id"]]
        assert len(ancestors) == 1
        cases.append({"case_index": case, "identity": identity, "NPZ_rows": rows,
            "raw_float32_payload_sha256": sha(np.ascontiguousarray(values["models"][rows[0]]).tobytes()),
            "raw_float64_sha256": f64_sha(raw_m), "local_derived_truth49_sha256": f64_sha(truth),
            "local_derived_initial49_sha256": f64_sha(initial), "initial_bound_margin": margin,
            "initial_m_minmax": [float(initial.min()), float(initial.max())],
            "old_linux_binding": old_binding,
            "local_truth_hash_matches_old": f64_sha(truth) == old_binding["truth_sha256"],
            "local_initial_hash_matches_old": f64_sha(initial) == old_binding["initial_native_m_sha256"],
            "ancestry_rows": ancestors})
    sources = [BASE/"source_selection.json", BASE/"dataset_manifest.json", ancestry_path,
        REPO/"results/provisional_software_benchmark/matched_cost_pilot/verification.py",
        REPO/"docs/ancestry_audit.md", OLD/"root_go.json", OLD/"protocol.json", OLD/"driver_source.py",
        OLD/"complete_receipt.json", NEW/"protocol.json", NEW/"FROZEN48_SCHEDULE.json"]
    return {"status": "RETAINED_VALIDATION_INPUT_SCHEMA_AND_ANCESTRY_BOUND_NO_PDE",
        "inspection_source_sha256": sha(Path(__file__).read_bytes()), "inputs": inputs,
        "NPZ_arrays": arrays, "NPZ_NPY_members": members, "cases": cases,
        "sources": {str(p.relative_to(ROOT)): {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size} for p in sources},
        "declared_original_source": {k: json.loads((BASE/"source_selection.json").read_bytes())[k]
            for k in ("dataset", "revision", "license", "scope", "raw_cache_retained")},
        "limits": "Read only validationNPZ and metadata/ancestry/prior receipts. Stored24grid/upper4 wavefields are finite/hash-bound, not PDE-validated or used for newobservations. Local49truth/initializer are source-informed interpolation/smoothing derivations, not new native-resolution geology. OldLinux all8 observation hashes are archived provenance, not newly reproduced. OriginalH5 source hashes are recorded metadata, not reread rawpayloads. Eight reused slices/four nominalIDs, SEAMIDs share oneancestor; no IID/site/physical/recovery/novelty/GO claim.",
        "runtime": {"python": sys.version, "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy")},
        "actions": {"PDE_calls": 0, "native_owners": 0, "jobs": 0, "downloads": 0, "production_edits": 0,
            "frozen_API_edits": 0}, "scientific_goals_complete": False}


if __name__ == "__main__":
    result = inspect()
    output = Path(__file__).with_name("RETAINED_INPUTS_01.json")
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "report_sha256": sha(output.read_bytes()),
        "cases": len(result["cases"]), "initial_hash_matches_old": sum(c["local_initial_hash_matches_old"] for c in result["cases"]),
        "truth_hash_matches_old": sum(c["local_truth_hash_matches_old"] for c in result["cases"]), "PDE_calls": 0}))
