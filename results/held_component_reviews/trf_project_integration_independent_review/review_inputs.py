"""Read-only stdlib reproduction of pinned input/schema/provenance, never physics."""
from pathlib import Path
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "work/new-phase-staging/trf_project_integration_design_candidate"
OLD = ROOT / "work/prior48_remote_study_01/extracted/study"
REPO = ROOT / "work/wavecert-fwi"
CHECKS = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(name, condition):
    CHECKS.append({"name": name, "pass": bool(condition)})
    if not condition:
        raise ValueError(name)


def read_json(path):
    return json.loads(path.read_bytes())


def decode_npz(path, expected, pinned_sha, pinned_size):
    raw = path.read_bytes()
    check(path.name + ": exact container", len(raw) == pinned_size and sha(raw) == pinned_sha)
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        check(path.name + ": exact unique members", len(infos) == len(expected)
              and {i.filename for i in infos} == {key + ".npy" for key in expected})
        for info in infos:
            key = info.filename[:-4]
            shape, descr = expected[key]
            element_bytes = {"<f8": 8, "<f4": 4, "<i8": 8}[descr]
            payload_bytes = math.prod(shape) * element_bytes
            cap = payload_bytes + 4112
            check(path.name + ": bounded member " + key,
                  not info.flag_bits & 1 and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                  and info.file_size <= cap)
            with archive.open(info) as stream:
                member = stream.read(cap + 1)
                check(path.name + ": bounded actual inflation " + key, len(member) <= cap and not stream.read(1))
            check(path.name + ": NPY magic " + key, member[:6] == b"\x93NUMPY")
            version = tuple(member[6:8])
            check(path.name + ": NPY version " + key, version in ((1, 0), (2, 0)))
            prefix = 10 if version == (1, 0) else 12
            header_len = struct.unpack("<H" if prefix == 10 else "<I", member[8:prefix])[0]
            check(path.name + ": bounded NPY header " + key, 0 < header_len <= 4096)
            header_raw = member[prefix:prefix + header_len]
            check(path.name + ": exact header extent " + key, len(header_raw) == header_len and header_raw.endswith(b"\n"))
            header = ast.literal_eval(header_raw.decode("latin1"))
            check(path.name + ": exact safe NPY schema " + key,
                  type(header) is dict and set(header) == {"shape", "descr", "fortran_order"}
                  and type(header["shape"]) is tuple and all(type(n) is int and n >= 0 for n in header["shape"])
                  and header["shape"] == shape and header["descr"] == descr and header["fortran_order"] is False)
            body = member[prefix + header_len:]
            check(path.name + ": exact payload and EOF " + key, len(body) == payload_bytes)
            code = {"<f8": "<d", "<f4": "<f", "<i8": "<q"}[descr]
            values = [v[0] for v in struct.iter_unpack(code, body)]
            check(path.name + ": finite payload " + key, all(math.isfinite(v) for v in values))
            result[key] = {"shape": shape, "descr": descr, "body": body, "values": values,
                           "payload_sha256": sha(body), "NPY_sha256": sha(member), "NPY_bytes": len(member)}
    return result


def main():
    inventory = read_json(STAGE / "STAGED_FILE_MANIFEST.json")
    check("exact design inventory pin", sha((STAGE / "STAGED_FILE_MANIFEST.json").read_bytes())
          == "180c997bf982679f3dcfdc49d98395a85c348e1b4670f99b45936bdc39d0bc59")
    check("12 files 214823 bytes", len(inventory["files"]) == 12
          and sum(row["bytes"] for row in inventory["files"].values()) == 214823)
    for name, row in inventory["files"].items():
        path = STAGE / name
        check("design inventory: " + name, path.is_file() and not path.is_symlink()
              and path.stat().st_size == row["bytes"] and sha(path.read_bytes()) == row["sha256"])
    plan = read_json(STAGE / "PROJECT_INPUTS_AND_WORKER_PLAN_01.json")
    inputs = read_json(STAGE / "RETAINED_INPUTS_01.json")
    initial_receipt = read_json(STAGE / "FROZEN_INITIAL_INPUT_RECEIPT.json")
    prior = read_json(OLD / "complete_receipt.json")
    for name, row in inputs["sources"].items():
        path = ROOT / name
        check("bound retained source: " + name, path.is_file()
              and path.stat().st_size == row["bytes"] and sha(path.read_bytes()) == row["sha256"])
    for name, row in inputs["inputs"].items():
        path = REPO / name
        check("bound project input: " + name, path.stat().st_size == row["bytes"]
              and sha(path.read_bytes()) == row["sha256"])
    arrays = decode_npz(STAGE / "frozen_initials.npz", {
        "initial_m": ((8, 2401), "<f8"), "case_indices": ((8,), "<i8"), "shape": ((2,), "<i8")},
        initial_receipt["npz_sha256"], initial_receipt["npz_bytes"])
    for key, array in arrays.items():
        check("initializer payload identity: " + key,
              array["payload_sha256"] == initial_receipt["schema"][key]["payload_sha256"]
              == plan["concrete_inputs"]["frozen_initializer_input"]["schema"][key]["payload_sha256"])
    check("initializer case indices", arrays["case_indices"]["values"] == list(range(8)))
    check("initializer physical shape", arrays["shape"]["values"] == [49, 49])
    check("initializer interior common box", all(1/36 + 1e-10 < v < 1 - 1e-10
          for v in arrays["initial_m"]["values"]))
    selected = {row["case_index"]: row for row in initial_receipt["case_provenance"]}
    check("exact8 provenance cases", set(selected) == set(range(8)))
    row_bytes = 2401 * 8
    for row in prior["runs"]:
        case = row["case_index"]
        raw = (OLD / row["return_artifact"]).read_bytes()
        check("retained return hash: " + row["return_artifact"], sha(raw) == row["return_sha256"])
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            decoded = stream.read(4 * 1024 * 1024 + 1)
            check("bounded retained return: " + row["return_artifact"], len(decoded) <= 4 * 1024 * 1024 and not stream.read(1))
        payload = json.loads(decoded)
        vector = payload["initial_model"]
        check("retained initializer schema: " + row["return_artifact"], type(vector) is list
              and len(vector) == 2401 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in vector))
        body = struct.pack("<2401d", *vector)
        check("all6 old initials exact: " + row["return_artifact"],
              body == arrays["initial_m"]["body"][case * row_bytes:(case + 1) * row_bytes]
              and sha(body) == row["common_binding"]["initial_native_m_sha256"]
              == selected[case]["initial_m_C_float64_sha256"])
        if row["return_artifact"] == selected[case]["return_artifact"]:
            check("chosen extraction provenance: " + str(case),
                  sha(raw) == selected[case]["return_artifact_sha256"] and row["identity"] == selected[case]["identity"])
    check("old exact artifact denominator", len(prior["artifact_sha256"]) == 107)
    check("no old NPY NPZ artifact", not any(name.endswith((".npy", ".npz")) for name in prior["artifact_sha256"]))
    raw_schema = {key: (tuple(row["shape"]), {"float32": "<f4", "float64": "<f8", "int64": "<i8"}[row["dtype"]])
                  for key, row in inputs["NPZ_arrays"].items()}
    raw_input = REPO / "data/reference/subsurfacegen_parent_split/validation.npz"
    raw_arrays = decode_npz(raw_input, raw_schema, "36d11aba94ae9093cac888e9b28a2d01edcef1aa3a54b12f8e10269b963db2fc", 842712)
    for key, array in raw_arrays.items():
        check("raw exact array payload: " + key, array["payload_sha256"] == inputs["NPZ_arrays"][key]["C_payload_sha256"]
              == plan["concrete_inputs"]["raw_NPZ_schema"][key]["C_payload_sha256"])
        check("raw exact NPY member: " + key, array["NPY_sha256"] == inputs["NPZ_NPY_members"][key + ".npy"]["NPY_sha256"])
    check("raw24 shape", raw_arrays["shape"]["values"] == [24, 24])
    check("raw24 spacing", raw_arrays["spacing"]["values"] == [.05])
    check("raw20 receiver slots", raw_arrays["receiver_indices"]["values"] == list(range(50, 70)))
    metadata = list(csv.DictReader(io.StringIO((REPO / "data/reference/subsurfacegen_parent_split/validation_metadata.csv").read_text())))
    old_protocol = read_json(OLD / "protocol.json")
    pairs = {(s, f) for s in old_protocol["original_geometry_context"]["original_source_indices"]
             for f in old_protocol["original_geometry_context"]["frequencies_hz"]}
    check("raw 192 metadata rows", len(metadata) == 192)
    for case in inputs["cases"]:
        identity = case["identity"]
        rows = [i for i, row in enumerate(metadata) if row["slice_id"] == identity["slice_id"]]
        check("24 repeats and metadata: " + str(case["case_index"]), len(rows) == 24 and rows == case["NPZ_rows"]
              and rows[0] == identity["array_index"] and all(metadata[i]["source_sha256"] == identity["source_sha256"]
              and metadata[i]["source_file"] == identity["source_file"] and metadata[i]["model_id"] == identity["parent_id"]
              and metadata[i]["split"] == "validation" for i in rows))
        check("6x4 geometry: " + str(case["case_index"]),
              {(raw_arrays["source_indices"]["values"][i], raw_arrays["frequencies_hz"]["values"][i]) for i in rows} == pairs)
        body = raw_arrays["models"]["body"][rows[0] * 576 * 4:(rows[0] + 1) * 576 * 4]
        check("24 identical raw models: " + str(case["case_index"]), all(
              raw_arrays["models"]["body"][i * 576 * 4:(i + 1) * 576 * 4] == body for i in rows))
        f64 = struct.pack("<576d", *(v[0] for v in struct.iter_unpack("<f", body)))
        check("raw f32/f64 old binding: " + str(case["case_index"]), sha(body) == case["raw_float32_payload_sha256"]
              and sha(f64) == case["raw_float64_sha256"] == case["old_linux_binding"]["raw_model_sha256"])
    ancestry = {row["base_volume_ancestor_id"] for case in inputs["cases"] for row in case["ancestry_rows"]}
    check("three declared ancestry clusters", ancestry == set(plan["concrete_inputs"]["ancestry"]["known_base_ancestor_clusters"]))
    check("four nominal parent labels", {case["identity"]["parent_id"] for case in inputs["cases"]}
          == set(plan["concrete_inputs"]["ancestry"]["nominal_parent_labels"]))
    for name, row in inventory["files"].items():
        check("unchanged after read-only review: " + name, sha((STAGE / name).read_bytes()) == row["sha256"])
    result = {"status": "PINNED_INPUT_STD_LIB_PROVENANCE_SCHEMA_REVIEW_CLEAR_NO_PDE",
              "review_script_sha256": sha(Path(__file__).read_bytes()), "checks": CHECKS,
              "check_count": len(CHECKS), "issues": [], "source_inventory_sha256": sha((STAGE / "STAGED_FILE_MANIFEST.json").read_bytes()),
              "decoded_NPZ_members": 10, "retained_initials_compared": 48,
              "recomputed_truth_or_gaussian_initializers": 0,
              "array_scope": "Stdlib bounded read of exact pinned initial/raw containers, basic finite/schema/hash and byte equality only. No PDE, interpolation, Gaussian filter, native imports, producer imports or test execution.",
              "scientific_goals_complete": False}
    output = Path(__file__).with_name("READONLY_INPUT_BINDING_01.json")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"check_count": len(CHECKS), "issues": [], "report_sha256": sha(output.read_bytes()), "path": str(output)}))


if __name__ == "__main__":
    main()
