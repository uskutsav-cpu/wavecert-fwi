"""Audit saved partial-plane transport/bytes only; no network or producer imports."""

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

LEDGER_SHA = "babbf4ec52bc6be2b54efc792a37cfbfba1c49ec8cd6d003a27d401caa056aa5"
PROPOSAL_SHA = "f2cea059ad3c116ad08131d7b874eab193f30929297f3068544bbd5fa64a64e6"
DRIVER_SHA = "e78c3f185d912c39d223c9414093a2f1831b3a30a09331571b6f7ac9c81c8e9f"
REVIEW_SHA = "d0499c4b387ec8955e3aecb910840eac6b9c81ff33e2775b45c5ef7949fb2ad2"
FILE_URL = "https://zenodo.org/api/records/4252588/files/overthrust_3D_true_model.h5/content"
RECORD_URL = "https://zenodo.org/api/records/4252588"
FILE_SIZE = 531253092


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(args):
    count = 0
    failures = []
    bound = {}

    def check(label, condition):
        nonlocal count
        count += 1
        if not condition:
            failures.append(label)

    def read(path, expected=None):
        raw = path.read_bytes()
        actual = sha(raw)
        bound[str(path)] = actual
        if expected is not None:
            check(str(path) + " exact SHA", actual == expected)
        return raw

    def local(relative):
        path = (args.run / relative).resolve()
        if not path.is_relative_to(args.run.resolve()):
            raise ValueError("artifact path escapes acquisition output")
        return path

    complete = json.loads(read(args.run / "complete_receipt.json"))
    ledger = json.loads(read(args.run / "REQUEST_LEDGER.json", LEDGER_SHA))
    proposal = json.loads(read(args.run / "provenance/FULL207_MIDDLE400_PARTIAL_ACQUISITION_PROPOSAL.json", PROPOSAL_SHA))
    read(args.run / "provenance/acquire_partial_plane.py", DRIVER_SHA)
    review = json.loads(read(args.run / "provenance/INDEPENDENT_GEOMETRY_REVIEW.json", REVIEW_SHA))
    check("complete proposal/driver/ledger pins", complete["proposal_sha256"] == PROPOSAL_SHA and complete["driver_sha256"] == DRIVER_SHA and complete["request_ledger_sha256"] == LEDGER_SHA)
    check("complete status", complete["status"] == "COMPLETE_PARTIAL_RAW_ANONYMOUS_PLANE_ONLY" and ledger["status"] == "complete")
    check("no false terminal/wholefile claim", complete["actual_terminal_exit"] is None and complete["whole_file_checksum_verified"] is False and ledger["whole_file_checksum_verified"] is False)
    check("no physical/site/confirmation claim", complete["physics_acceptance"] is None and complete["independent_site_claim"] is False and complete["untouched_confirmation_claim"] is False)
    check("anonymous unchanged representation", ledger["arithmetic_conversion"] is False and ledger["axis_transpose"] is False and ledger["water_rows_removed"] == 0 and ledger["model_physical_units_verified"] is False and ledger["named_axis_order_verified"] is False)
    check("fixed selection", proposal["selection"] == {"dataset": "m", "stored_dtype": "<f4", "stored_shape": [207, 801, 801], "index": ["0:207", 400, "0:801"], "result_shape": [207, 801], "index_basis": "anonymous stored dimensions, C contiguous", "water_rows_removed": 0, "arithmetic_conversion": None, "axis_transpose": None})
    expected_plan = []
    for row in range(207):
        lo = 2192 + 4 * (row * 801 * 801 + 400 * 801)
        expected_plan.append({"index0": row, "first_byte": lo, "last_byte": lo + 3203, "body_bytes": 3204})
    check("independent integer range plan", proposal["intervals"] == expected_plan)
    check("fixed body cap and plane count", proposal["hard_response_body_cap_bytes"] == ledger["body_cap_bytes"] == 8388608 and proposal["planned_plane_response_body_bytes"] == 663228)
    file_identity = {"id": "436595b8-ed4b-4e1d-8991-e63608dcc169", "key": "overthrust_3D_true_model.h5", "size": FILE_SIZE, "checksum": "md5:ae4082cdcebbc7714eedd382d7925804", "links": {"self": FILE_URL}}
    check("fixed published file identity claim", proposal["file"] == file_identity)
    for relative, expected in review["source_assets"].items():
        provenance = read(local("provenance/" + relative), expected)
        cached = read((args.cache / relative).resolve(), expected)
        check("copied reviewed source " + relative, provenance == cached)
    metadata = []
    for prefix, ledger_name in (("", "REQUEST_LEDGER.json"), ("geometry_addendum/", "REQUEST_LEDGER_GEOMETRY.json")):
        source_ledger = json.loads(read(local("provenance/" + prefix + ledger_name)))
        for row in source_ledger["requests"]:
            interval = row.get("requested_range")
            if interval is not None:
                lo, hi = interval
                check("metadata excludes model storage", 0 <= lo <= hi < 2192 or 531247820 <= lo <= hi < FILE_SIZE)
                metadata.append((lo, hi, prefix + row["local_name"], row["sha256"]))
    check("25 cached metadata ranges", len(metadata) == 25)
    schedule = [("before_record", None)]
    schedule += [(f"before_metadata_{index}", (lo, hi)) for index, (lo, hi, _, _) in enumerate(metadata)]
    schedule += [(f"model_row_{row['index0']:03d}", (row["first_byte"], row["last_byte"])) for row in expected_plan]
    schedule += [("after_record", None)]
    schedule += [(f"after_metadata_{index}", (lo, hi)) for index, (lo, hi, _, _) in enumerate(metadata)]
    check("259 requests fixed ordered schedule", len(ledger["requests"]) == len(schedule) == 259 and [(r["label"], tuple(r["requested_range"]) if r["requested_range"] is not None else None) for r in ledger["requests"]] == schedule)
    body_total = 0
    model_total = 0
    record_etags = []
    file_etags = []
    for request, (label, interval) in zip(ledger["requests"], schedule, strict=True):
        headers = request["response_headers"]
        check(label + " completed response", request["status"] == "complete")
        expected_url = RECORD_URL if interval is None else FILE_URL
        check(label + " exact endpoint", request["url"] == request["final_url"] == expected_url)
        check(label + " identity encoding", headers["Content-Encoding"] in (None, "identity"))
        check(label + " nonnegative finite duration", math.isfinite(request["elapsed_seconds"]) and request["elapsed_seconds"] >= 0)
        if interval is None:
            check(label + " HTTP200 JSON", request["HTTP_status"] == 200 and headers["Content-Range"] is None)
            phase = label.split("_")[0]
            raw = read(args.run / f"metadata/{phase}_record.json", request["sha256"])
            record = json.loads(raw)
            files = [r for r in record["files"] if r["key"] == file_identity["key"]]
            check(label + " published identity and license", record["id"] == 4252588 and record["revision"] == 2 and files == [file_identity] and record["metadata"]["license"]["id"] == "cc-by-4.0")
            check(label + " exact cached record", sha(raw) == proposal["record"]["cached_record_sha256"] and raw == read(args.run / "provenance/zenodo_4252588_record.json"))
            check(label + " bounded JSON", request["maximum_body_bytes"] == 65536 and len(raw) <= 65536)
            record_etags.append(headers["ETag"])
        else:
            lo, hi = interval
            length = hi - lo + 1
            check(label + " HTTP206 exact ContentRange", request["HTTP_status"] == 206 and headers["Content-Range"] == f"bytes {lo}-{hi}/{FILE_SIZE}")
            check(label + " exact admitted length", request["maximum_body_bytes"] == length)
            if label.startswith("model_row_"):
                row = int(label.rsplit("_", 1)[1])
                raw = read(args.run / f"rows/row_{row:03d}.bin", request["sha256"])
                check(label + " exact model row size", len(raw) == 3204)
                model_total += len(raw)
            else:
                phase, _, index_text = label.split("_")
                index = int(index_text)
                raw = read(args.run / f"metadata/{phase}_{index}_{lo}_{hi}.bin", request["sha256"])
                _, _, source, expected = metadata[index]
                check(label + " exact cached metadata", sha(raw) == expected and raw == read(args.run / ("provenance/" + source)))
            file_etags.append(headers["ETag"])
        check(label + " actual body and declared length", len(raw) == request["body_bytes_read"] == int(headers["Content-Length"]))
        body_total += len(raw)
    check("all body counts", body_total == ledger["body_bytes_read"] == 686662 and body_total <= 8388608 and model_total == ledger["model_response_body_bytes"] == 663228)
    check("207 complete row entries", len(ledger["complete_rows"]) == ledger["model_rows_complete"] == 207 and [r["index0"] for r in ledger["complete_rows"]] == list(range(207)))
    chunks = []
    for index, row in enumerate(ledger["complete_rows"]):
        check(f"row{index} exact path", row["path"] == f"rows/row_{index:03d}.bin")
        chunks.append(read(local(row["path"]), row["sha256"]))
    payload = b"".join(chunks)
    check("ordered raw row assembly", len(payload) == 663228 and sha(payload) == ledger["ordered_raw_plane_sha256"])
    npy = read(args.run / "anonymous_m_axis1_400.npy")
    check("NPY v1 magic", npy[:8] == b"\x93NUMPY\x01\x00")
    header_length = struct.unpack("<H", npy[8:10])[0]
    header = ast.literal_eval(npy[10:10 + header_length].decode("ascii").strip())
    check("NPY header representation", header == {"descr": "<f4", "fortran_order": False, "shape": (207, 801)})
    check("NPY payload exact rows without conversion", npy[10 + header_length:] == payload and len(npy) == 663308)
    value_count = 0
    finite_positive = True
    for value, in struct.iter_unpack("<f", payload):
        value_count += 1
        finite_positive = finite_positive and math.isfinite(value) and value > 0
    check("all165807 stored values finite positive", value_count == 207 * 801 == 165807 and finite_positive)
    check("artifact manifest complete membership", {str(p.relative_to(args.run)) for p in args.run.rglob("*") if p.is_file() and p.name != "complete_receipt.json"} == set(complete["artifacts"]))
    for relative, expected in complete["artifacts"].items():
        read(local(relative), expected)
    check("all bound files unchanged", all(sha(Path(path).read_bytes()) == expected for path, expected in bound.items()))
    findings = []
    if any(value is not None for value in record_etags):
        findings.append({"kind": "scope_wording", "finding": "The producer's blanket no-ETag statement is too broad: record-JSON responses carry an ETag, whereas HDF5 file-range responses do not. A record revision ETag does not identify the whole HDF5 byte stream.", "record_ETags": record_etags, "HDF5_file_ETags": sorted(set(file_etags), key=str), "raw_history_unchanged": True})
    report = {
        "status": "INTEGRITY_FAIL" if failures else "INTEGRITY_PASS_WITH_DISCLOSURE_FINDING" if findings else "INTEGRITY_PASS",
        "checks": count, "integrity_failures": failures, "disclosure_findings": findings,
        "source_sha256": sha(Path(__file__).read_bytes()), "bound_file_sha256": bound,
        "request_count": len(ledger["requests"]), "complete_rows": 207,
        "HTTP206_file_range_responses": len(file_etags), "HTTP200_record_responses": len(record_etags),
        "actual_total_response_body_bytes": body_total, "actual_model_response_body_bytes": model_total,
        "ordered_raw_plane_sha256": sha(payload), "NPY_sha256": sha(npy),
        "stored_array_shape": [207, 801], "stored_dtype": "<f4", "stored_values_checked_finite_positive": value_count,
        "network_requests_by_auditor": 0, "producer_imports": 0, "PDE_calls": 0,
        "arithmetic_conversion": False, "transpose": False, "row_drop": False,
        "whole_file_checksum_verified": False, "physical_units_verified": False, "named_axes_verified": False,
        "physical_acceptance": None, "independent_site_claim": False, "untouched_confirmation_claim": False,
        "scope": "Independent audit of saved local bytes and response ledger, not independent refetch/authentication, original CD1 velocity identity, whole-HDF5 immutability, validated physical units, geology/physics confirmation, or inversion success. Root actual process terminal observation is separate; producer receipt correctly leaves terminal exit null.",
    }
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": count, "integrity_failures": len(failures), "disclosure_findings": len(findings), "report_sha256": sha(args.report.read_bytes())}))
    return 2 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "cache", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    return audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
