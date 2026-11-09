"""Offline independent check of cached metadata; never opens network/model data."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FILE_SIZE = 531253092
M_START, M_STOP = 2192, 531247820


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages(path: Path) -> dict[int, bytes]:
    raw = path.read_bytes()
    offset = 0
    result = {}
    while offset < len(raw):
        kind, size, _flags = struct.unpack_from("<HHB", raw, offset)
        assert size % 8 == 0 and offset + 8 + size <= len(raw)
        if kind:
            assert kind not in result
            result[kind] = raw[offset + 8: offset + 8 + size]
        offset += 8 + size
    assert offset == len(raw)
    return result


def main() -> None:
    checks = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise ValueError(label)
        checks.append(label)

    base_ledger = json.loads((BASE / "REQUEST_LEDGER.json").read_bytes())
    geom_ledger = json.loads((HERE / "REQUEST_LEDGER_GEOMETRY.json").read_bytes())
    check(geom_ledger["prior_request_ledger_sha256"] == digest(BASE / "REQUEST_LEDGER.json"), "prior ledger binding")
    check(geom_ledger["body_bytes_prior"] == base_ledger["body_bytes_read"] == 15504, "prior body count")
    ranges = []
    counted = 0
    for folder, ledger in ((BASE, base_ledger), (HERE, geom_ledger)):
        local_total = 0
        for i, entry in enumerate(ledger["requests"]):
            prefix = f"{folder.name} request {i}"
            nbytes = entry["body_bytes_read"]
            local_total += nbytes
            if entry["kind"] == "head":
                check(nbytes == 0 and entry["status"] == "HEAD_only_no_body_read", prefix + " HEAD body zero")
                continue
            check(entry["status"] == "bounded_metadata_saved", prefix + " complete")
            path = folder / entry["local_name"]
            check(path.stat().st_size == nbytes, prefix + " length")
            check(digest(path) == entry["sha256"], prefix + " SHA256")
            if entry.get("requested_range") is not None:
                lo, hi = entry["requested_range"]
                check(0 <= lo <= hi < FILE_SIZE, prefix + " file bounds")
                check(hi < M_START or lo >= M_STOP, prefix + " excludes all model storage")
                check(entry["HTTP_status"] == 206, prefix + " status 206")
                check(entry["response_headers"]["Content-Range"] == f"bytes {lo}-{hi}/{FILE_SIZE}", prefix + " exact Content-Range")
                check(int(entry["response_headers"]["Content-Length"]) == nbytes == hi - lo + 1, prefix + " exact response length")
                check(entry["response_headers"]["Content-Encoding"] in (None, "identity"), prefix + " identity body")
                ranges.append([lo, hi])
        expected = ledger["body_bytes_read"] - ledger.get("body_bytes_prior", 0)
        check(local_total == expected, folder.name + " body sum")
        counted += local_total
    check(counted == geom_ledger["body_bytes_read"] == 19864 < 262144, "combined metadata body cap")
    check(len(ranges) == 25, "25 metadata intervals only")

    m = messages(BASE / "model_m_object_header_messages_1480_1735.bin")
    shape = struct.unpack_from("<3Q", m[1], 8)
    check(shape == (207, 801, 801), "m dataspace shape")
    check(m[3].hex() == "11201f000400000000002000170800177f00000000000000", "m IEEE float32 little-endian datatype")
    check(m[8][:2] == b"\x03\x01", "m contiguous layout")
    check(struct.unpack_from("<QQ", m[8], 2) == (M_START, M_STOP-M_START), "m raw storage span")
    check(math.prod(shape) * 4 == M_STOP - M_START, "m shape bytes")
    check(11 not in m and 12 not in m, "m object has no local filter or attribute messages")

    geometry = {}
    plans = json.loads((HERE / "GEOMETRY_SCALAR_READ_FREEZE.json").read_bytes())["plans"]
    claims = {x["name"]: x for x in json.loads((HERE / "VERIFIED_GEOMETRY_SCALARS.json").read_bytes())}
    header_names = {"n": "n_object_messages_816_1071.bin", "o": "o_object_messages_1160_1415.bin", "d": "d_object_messages_1800_2055.bin", "copyright": "copyright_object_messages_531247860_531248115.bin"}
    for plan in plans:
        name = plan["name"]
        header = messages(HERE / header_names[name])
        offset, length = struct.unpack_from("<QQ", header[8], 2)
        check(offset == plan["first_byte"] and length == plan["body_bytes"], name + " value span bound to raw object header")
        check(offset + length == plan["last_byte"] + 1, name + " halfopen span")
        check(offset + length <= M_START or offset >= M_STOP, name + " raw values exclude m")
        filename = f"geometry_{name}_values_{offset}_{offset+length-1}.bin"
        raw = (HERE / filename).read_bytes()
        check(len(raw) == length and digest(HERE / filename) == claims[name]["source_body_sha256"], name + " value bytes identity")
        if name in ("n", "d"):
            check(header[3].hex() == "10080000080000000000400000000000", name + " signed little-endian int64")
            value = list(struct.unpack("<3q", raw))
        elif name == "o":
            check(header[3].hex() == "11203f000800000000004000340b0034ff03000000000000", "o IEEE float64 little-endian")
            value = list(struct.unpack("<3d", raw))
        else:
            check(header[3].hex() == "13100000800c0000", "copyright fixed 3200-byte string")
            value = raw.decode("ascii")
        check(value == claims[name]["value"], name + " independent decode agrees")
        geometry[name] = value
    check(geometry["n"] == [801, 801, 207] and geometry["n"][::-1] == list(shape), "n reverse matches m shape")
    check(geometry["d"] == [25, 25, 25] and geometry["o"] == [0.0, 0.0, 0.0], "d and o values")
    check((HERE / "HDF5_COPYRIGHT_METADATA.txt").read_bytes().decode("ascii") == geometry["copyright"], "copyright transcription exact")

    record_path = BASE / "zenodo_4252588_record.json"
    record = json.loads(record_path.read_bytes())
    candidates = [x for x in record["files"] if x["key"] == "overthrust_3D_true_model.h5"]
    check(len(candidates) == 1, "one named primary true model")
    file = candidates[0]
    check(file["id"] == "436595b8-ed4b-4e1d-8991-e63608dcc169" and file["size"] == FILE_SIZE, "record file UUID and size")
    check(file["checksum"] == "md5:ae4082cdcebbc7714eedd382d7925804", "published whole-file checksum claim")
    check(record["id"] == 4252588 and record["revision"] == 2 and record["metadata"]["license"]["id"] == "cc-by-4.0", "record version revision and license declaration")

    intervals = []
    for row in range(shape[0]):
        start = M_START + ((row * shape[1] + 400) * shape[2]) * 4
        stop = start + shape[2]*4
        check(M_START <= start < stop <= M_STOP, f"planned row {row} within model storage")
        intervals.append({"index0": row, "first_byte": start, "last_byte": stop-1, "body_bytes": stop-start})
    old = json.loads((BASE / "UNEXECUTED_SELECTED_PLANE_RANGE_PLANS.json").read_bytes())["plans"][0]
    check(intervals == old["intervals"], "independent full207 range plan agrees with prior anonymous plan")
    total = sum(x["body_bytes"] for x in intervals)
    check(total == 663228 < 8388608, "selected plane below 8 MiB")
    proposal = {
        "status": "UNEXECUTED_PROPOSAL_REQUIRES_ROOT_SOURCE_REVIEW_AND_FREEZE",
        "model_values_read_during_this_review": False,
        "record": {"id": 4252588, "revision": 2, "doi": "10.5281/zenodo.4252588", "cached_record_sha256": digest(record_path)},
        "file": file,
        "whole_file_checksum_verified": False,
        "hard_response_body_cap_bytes": 8388608,
        "planned_plane_response_body_bytes": total,
        "selection": {"dataset": "m", "stored_dtype": "<f4", "stored_shape": list(shape), "index": ["0:207", 400, "0:801"], "result_shape": [207, 801], "index_basis": "anonymous stored dimensions, C contiguous", "water_rows_removed": 0, "arithmetic_conversion": None, "axis_transpose": None},
        "intervals": intervals,
        "required_pre_body_guards": [
            "Re-fetch bounded versioned record JSON and require same record id/revision, true-file UUID, name, size, checksum claim and content URL; fail if changed.",
            "Require exact reviewed range plan and total before network; no retry or resume on partial cohort; create fresh exclusive output directory.",
            "Set Accept-Encoding identity and exact Range bytes=lo-hi. Reject HTTP status other than 206 before reading any body, including HTTP 200 or error bodies.",
            "Require final HTTPS host zenodo.org and exact pinned path; reject redirects to other file or record identity.",
            "Require exact Content-Range bytes lo-hi/531253092 and Content-Length 3204 for every model row; reject encoded responses.",
            "Record every attempted request and count actual body bytes toward aggregate 8 MiB cap, including incomplete/error bodies if any are consumed; never read a rejected header response body.",
            "Read exactly the admitted length with bounded chunks and deadline; require completed 3204-byte body; no whole-file fallback, external HDF5 automatic fetching, decompression, or broad range coalescing.",
            "Bind captured metadata header/value files to reviewed SHA256s; repeat small header/value ranges and compare exact bytes before the first and after the last row. Count these in the same cap.",
            "Save each raw row separately and SHA256, ordered concatenated row bytes SHA256, request headers/status/timings/body counts and actual terminal exit; never claim whole-file checksum verification from a subset.",
            "Mark any interrupted or failed cohort incomplete and exclude from valid dataset credit. Preserve its bytes/ledger; a later fresh acquisition is a separately registered cohort.",
            "Perform no value-driven slice selection, row dropping, clipping, smoothing, unit conversion, velocity reconstruction, acquisition-physics run or confirmation claim in acquisition.",
            "Carry primary record CC BY 4.0 statement, attribution, original copyright bytes, link, accessed/version identities and explicit partial-selection/change notice with future dataset outputs."
        ],
        "interpretation": {"n": geometry["n"], "d": geometry["d"], "o": geometry["o"], "named_axis_order_verified": False, "m_physical_units_verified": False, "d_physical_units_explicit_in_hdf5_metadata": False, "water_row_claim_verified": False, "original_CD1_velocity_bit_identity_verified": False},
        "not_achieved": ["model acquisition", "independent physical units or axis interpretation", "original CD1 velocity identity", "realistic wave-physics acceptance", "independent geological sites", "untouched confirmation", "student final research design"]
    }
    (HERE / "FULL207_MIDDLE400_PARTIAL_ACQUISITION_PROPOSAL.json").write_text(json.dumps(proposal, indent=2)+"\n")
    receipt = {
        "status": "OFFLINE_METADATA_REVIEW_CLEAR_WITH_EXPLICIT_INTERPRETATION_LIMITS",
        "source_sha256": digest(Path(__file__)),
        "checks_count": len(checks), "checks": checks,
        "geometry_addendum_body_bytes": 4360, "all_metadata_body_bytes": counted,
        "model_byte_ranges_requested": 0, "model_values_decoded": 0, "network_requests_from_this_review": 0,
        "geometry_values": {k: geometry[k] for k in ("n", "d", "o")},
        "source_assets": {str(p.relative_to(BASE)): digest(p) for p in sorted(BASE.rglob("*")) if p.is_file() and p.suffix in (".bin",) or p.is_file() and p.name in ("REQUEST_LEDGER.json", "REQUEST_LEDGER_GEOMETRY.json", "GEOMETRY_SCALAR_READ_FREEZE.json", "VERIFIED_GEOMETRY_SCALARS.json", "zenodo_4252588_record.json", "HDF5_COPYRIGHT_METADATA.txt")},
        "proposal_sha256": digest(HERE / "FULL207_MIDDLE400_PARTIAL_ACQUISITION_PROPOSAL.json")
    }
    (HERE / "INDEPENDENT_GEOMETRY_REVIEW.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("status", "checks_count", "all_metadata_body_bytes", "model_values_decoded", "proposal_sha256")}))


if __name__ == "__main__":
    main()
